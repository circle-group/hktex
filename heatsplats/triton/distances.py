import torch
import triton
import triton.language as tl

__all__ = ["compute_biharmonic_distance_pairwise"]


@triton.jit
def _biharmonic_fwd_kernel(
    # Pointers to Tensors
    evecs_i_ptr,
    evecs_j_ptr,
    evals_ptr,
    output_ptr,
    # Dimensions
    I,
    J,
    K,
    # Strides for memory access
    stride_i_i,
    stride_i_k,
    stride_j_j,
    stride_j_k,
    stride_evals_k,
    stride_out_j,
    stride_out_i,
    # Kernel parameters
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Forward kernel for biharmonic distance.
    Each program computes one element of the (J, I) output matrix.
    """
    # Get the program ID for the (j, i) location in the output matrix
    pid_j = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointer to the start of the row for the current i-th and j-th point
    evecs_i_ptrs = evecs_i_ptr + pid_i * stride_i_i + offs_k * stride_i_k
    evecs_j_ptrs = evecs_j_ptr + pid_j * stride_j_j + offs_k * stride_j_k
    evals_ptrs = evals_ptr + offs_k * stride_evals_k

    # Accumulator for the squared distance
    acc = 0.0
    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K - k * BLOCK_SIZE_K
        # Load blocks of data
        e_i = tl.load(evecs_i_ptrs, mask=k_mask, other=0.0)
        e_j = tl.load(evecs_j_ptrs, mask=k_mask, other=0.0)
        evals = tl.load(evals_ptrs, mask=k_mask, other=1.0)

        # Fused computation for the current block
        diff = e_i - e_j
        # NOTE: Assumes evals are non-zero.
        weighted_sq_diff = (diff * diff) / (evals * evals)
        acc += tl.sum(weighted_sq_diff)

        evecs_i_ptrs += BLOCK_SIZE_K * stride_i_k
        evecs_j_ptrs += BLOCK_SIZE_K * stride_j_k
        evals_ptrs += BLOCK_SIZE_K * stride_evals_k

    # Final distance is the square root of the accumulated sum
    dist = tl.sqrt(acc)

    # Store the result
    output_offset = pid_j * stride_out_j + pid_i * stride_out_i
    tl.store(output_ptr + output_offset, dist)


@triton.jit
def _biharmonic_bwd_kernel(
    # Pointers to Tensors from forward pass
    evecs_i_ptr,
    evecs_j_ptr,
    evals_ptr,
    output_ptr,
    grad_output_ptr,
    # Pointers to Output Gradients
    grad_evecs_i_ptr,
    grad_evecs_j_ptr,
    grad_evals_ptr,
    # Dimensions
    I,
    J,
    K,
    # Strides
    stride_i_i,
    stride_i_k,
    stride_j_j,
    stride_j_k,
    stride_evals_k,
    stride_out_j,
    stride_out_i,
    stride_grad_i_i,
    stride_grad_i_k,
    stride_grad_j_j,
    stride_grad_j_k,
    stride_grad_evals_k,
    # Kernel parameters
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Backward kernel for biharmonic distance.
    This kernel has a 3D grid, computing gradients for a block of (J, I, K).
    """
    # Get program IDs
    pid_j = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    evecs_i_ptrs = evecs_i_ptr + pid_i * stride_i_i + offs_k * stride_i_k
    evecs_j_ptrs = evecs_j_ptr + pid_j * stride_j_j + offs_k * stride_j_k
    evals_ptrs = evals_ptr + offs_k * stride_evals_k

    grad_evecs_i_ptrs = (
        grad_evecs_i_ptr + pid_i * stride_grad_i_i + offs_k * stride_grad_i_k
    )
    grad_evecs_j_ptrs = (
        grad_evecs_j_ptr + pid_j * stride_grad_j_j + offs_k * stride_grad_j_k
    )
    grad_evals_ptrs = grad_evals_ptr + offs_k * stride_grad_evals_k

    # Common term in the gradient calculation
    dist = tl.load(output_ptr + pid_j * stride_out_j + pid_i * stride_out_i)
    grad_out = tl.load(grad_output_ptr + pid_j * stride_out_j + pid_i * stride_out_i)
    # Gradient w.r.t the squared distance
    grad_dist_sq = grad_out / (2.0 * dist + 1e-8)

    # Loop over K to compute and accumulate gradients
    # This loop is outside the main reduction axes for grads
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K - k * BLOCK_SIZE_K

        # Load all necessary data for this block
        e_i = tl.load(evecs_i_ptrs, mask=k_mask, other=0.0)
        e_j = tl.load(evecs_j_ptrs, mask=k_mask, other=0.0)
        evals = tl.load(evals_ptrs, mask=k_mask, other=1.0)

        # --- Compute gradient terms ---
        diff = e_i - e_j
        evals_sq = evals * evals

        # Gradient contribution to evecs
        grad_term = 2.0 * grad_dist_sq * diff / evals_sq

        # Gradient contribution to evals
        grad_evals_term = -2.0 * grad_dist_sq * (diff * diff) / (evals_sq * evals)

        # Atomically add contributions to the output gradient tensors
        # This handles the reduction/summation required by the backward pass
        tl.atomic_add(grad_evecs_i_ptrs, grad_term, mask=k_mask)
        tl.atomic_add(grad_evecs_j_ptrs, -grad_term, mask=k_mask)
        tl.atomic_add(grad_evals_ptrs, grad_evals_term, mask=k_mask)

        evecs_i_ptrs += BLOCK_SIZE_K * stride_i_k
        evecs_j_ptrs += BLOCK_SIZE_K * stride_j_k
        evals_ptrs += BLOCK_SIZE_K * stride_evals_k

        grad_evecs_i_ptrs += BLOCK_SIZE_K * stride_grad_i_k
        grad_evecs_j_ptrs += BLOCK_SIZE_K * stride_grad_j_k
        grad_evals_ptrs += BLOCK_SIZE_K * stride_grad_evals_k


class BiharmonicDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, evecs_i, evecs_j, evals):
        # Ensure inputs are contiguous
        evecs_i = evecs_i.contiguous()
        evecs_j = evecs_j.contiguous()
        evals = evals.contiguous()

        I, K = evecs_i.shape
        J, _ = evecs_j.shape

        # NOTE: Your original code produces (J, I). This is a valid convention.
        output = torch.empty((J, I), dtype=evecs_i.dtype, device=evecs_i.device)

        # Save tensors for backward pass
        ctx.save_for_backward(evecs_i, evecs_j, evals, output)

        # Launch the forward kernel
        grid = (J, I)
        _biharmonic_fwd_kernel[grid](
            evecs_i,
            evecs_j,
            evals,
            output,
            I,
            J,
            K,
            evecs_i.stride(0),
            evecs_i.stride(1),
            evecs_j.stride(0),
            evecs_j.stride(1),
            evals.stride(0),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE_K=32,  # Can be tuned
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()

        # Unpack saved tensors
        evecs_i, evecs_j, evals, output = ctx.saved_tensors
        I, K = evecs_i.shape
        J, _ = evecs_j.shape

        # Prepare output tensors for gradients, initialized to zero
        grad_evecs_i = torch.zeros_like(evecs_i)
        grad_evecs_j = torch.zeros_like(evecs_j)
        grad_evals = torch.zeros_like(evals)

        # Launch the backward kernel
        grid = (J, I)
        _biharmonic_bwd_kernel[grid](
            evecs_i,
            evecs_j,
            evals,
            output,
            grad_output,
            grad_evecs_i,
            grad_evecs_j,
            grad_evals,
            I,
            J,
            K,
            evecs_i.stride(0),
            evecs_i.stride(1),
            evecs_j.stride(0),
            evecs_j.stride(1),
            evals.stride(0),
            output.stride(0),
            output.stride(1),
            grad_evecs_i.stride(0),
            grad_evecs_i.stride(1),
            grad_evecs_j.stride(0),
            grad_evecs_j.stride(1),
            grad_evals.stride(0),
            BLOCK_SIZE_K=32,
        )

        # Return one gradient for each input of the forward function
        return grad_evecs_i, grad_evecs_j, grad_evals


compute_biharmonic_distance_pairwise = BiharmonicDistance.apply
