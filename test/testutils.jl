using RossbyWaveSpectrum: matrix_block, StructMatrix
using BlockArrays
using StructArrays

function matrix_subsample!(M_subsample::BlockArray{<:Real}, M::BlockArray{<:Real}, nr_M, nr, nℓ, nvariables)
    nparams = nr*nℓ
    @views for colind in 1:nvariables, rowind in 1:nvariables
        Mblock = matrix_block(M, rowind, colind)
        M_subsample_block = matrix_block(M_subsample, rowind, colind)
        for ℓ′ind in 1:nℓ, ℓind in 1:nℓ
            Mblock_ℓℓ′ = Mblock[Block(ℓind, ℓ′ind)]
            M_subsample_block_ℓℓ′ = M_subsample_block[Block(ℓind, ℓ′ind)]
            M_subsample_block_ℓℓ′ .= Mblock_ℓℓ′[axes(M_subsample_block_ℓℓ′)...]
        end
    end
    return M_subsample
end

function matrix_subsample!(M_subsample::StructArray{<:Complex,2}, M::StructArray{<:Complex,2}, args...)
    matrix_subsample!(M_subsample.re, M.re, args...)
    matrix_subsample!(M_subsample.im, M.im, args...)
    return M_subsample
end

blockwise_cmp(f, x, y) = f(x,y)
function blockwise_cmp(f, S1::StructMatrix{<:Complex}, S2::StructMatrix{<:Complex})
    blockwise_cmp(f, S1.re, S2.re) && blockwise_cmp(f, S1.im, S2.im)
end
function blockwise_cmp(f, B1::BlockMatrix{<:Real}, B2::BlockMatrix{<:Real})
    all(zip(blocks(B1), blocks(B2))) do (x, y)
        blockwise_cmp(f, x, y)
    end
end

blockwise_isequal(x, y) = blockwise_cmp(==, x, y)
blockwise_isapprox(x, y; kw...) = blockwise_cmp((x, y) -> isapprox(x, y; kw...), x, y)

function rossby_ridge_eignorm(λ, v, (A, B), m, nparams; ΔΩ_frac = 0)
    matchind = argmin(abs.(real(λ) .- RossbyWaveSpectrum.rossby_ridge(m; ΔΩ_frac)))
    vi = v[:, matchind];
    λi = λ[matchind]
    normsden = [norm(λi * @view(vi[i*nparams .+ (1:nparams)])) for i in 0:2]
    normsnum = [norm((A[i*nparams .+ (1:nparams), :]*vi - λi*B[i*nparams .+ (1:nparams), :]*vi)) for i in 0:2]
    [(d/norm(λi) > 1e-10 ? n/d : 0) for (n, d) in zip(normsnum, normsden)]
end
