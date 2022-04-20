using LinearAlgebra: BlasInt, checksquare, BLAS, LAPACK
using .LAPACK: chkfinite, chklapackerror

function allocate_eigen_cache(A::Matrix{Complex{T}}) where {T}
    n = checksquare(A)
    scale = similar(A, T, n)
    rconde = similar(A, T, n)
    rcondv = similar(A, T, n)
    rwork = Vector{T}(undef, 2n)

    scale, rconde, rcondv, rwork
end

function eigenCF64!(A::AbstractMatrix{ComplexF64};
        lams = similar(A, ComplexF64, size(A, 1)),
        vecs::AbstractMatrix{ComplexF64} = similar(A, ComplexF64, size(A)),
        cache = allocate_eigen_cache(A),
        sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby,
        )

    n = checksquare(A)
    scale, rconde, rcondv, rwork = cache

    balanc = 'B'
    sense = 'N'
    # if balanc ∉ ['N', 'P', 'S', 'B']
    #     throw(ArgumentError("balanc must be 'N', 'P', 'S', or 'B', but $balanc was passed"))
    # end
    # if sense ∉ ['N','E','V','B']
    #     throw(ArgumentError("sense must be 'N', 'E', 'V' or 'B', but $sense was passed"))
    # end

    chkfinite(A) # balancing routines don't support NaNs and Infs
    lda = max(1,stride(A,2))

    VL = similar(A, ComplexF64, 0, n)

    ilo = Ref{BlasInt}()
    ihi = Ref{BlasInt}()
    abnrm = Ref{Float64}()

    work = Vector{ComplexF64}(undef, 1)
    lwork = BlasInt(-1)
    info = Ref{BlasInt}()
    for i = 1:2  # first call returns lwork as work[1]
        ccall((BLAS.@blasfunc(zgeevx_), LAPACK.libblastrampoline), Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
               Ref{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt}, Ptr{ComplexF64},
               Ptr{ComplexF64}, Ref{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt},
               Ptr{BlasInt}, Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ptr{ComplexF64}, Ref{BlasInt},
               Ptr{Float64}, Ptr{BlasInt}, Clong, Clong, Clong, Clong),
               balanc, 'N', 'V', sense,
               n, A, lda, lams,
               VL, 1, vecs, max(1,n),
               ilo, ihi, scale, abnrm,
               rconde, rcondv, work, lwork,
               rwork, info, 1, 1, 1, 1)
        chklapackerror(info[])
        if i == 1
            lwork = BlasInt(work[1])
            resize!(work, lwork)
        end
    end

    # A, lams, VL, vecs, ilo[], ihi[], scale, abnrm[], rconde, rcondv
    LinearAlgebra.Eigen(LinearAlgebra.sorteig!(lams, vecs, sortby)...)
end