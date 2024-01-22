using Documenter
using RossbyWaveSpectrum
using RossbyPlots

DocMeta.setdocmeta!(RossbyWaveSpectrum, :DocTestSetup, :(using RossbyWaveSpectrum); recursive=true)

makedocs(
    sitename = "RossbyWaveSpectrum",
    modules = [RossbyWaveSpectrum, RossbyPlots],
    authors = "Jishnu Bhattacharya",
    warnonly = :missing_docs,
    pages = [
        "Theory" => "theory.md",
        "Using the code" => "index.md",
        "Library" => "API.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/jishnub/RossbyWaveSpectrum.jl.git",
    push_preview = true,
)
