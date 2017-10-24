using Documenter, RGBDSegmentation

makedocs(
    format = :html,
    sitename = "RGBDSegmentatione",
    pages = [
        "index.md",
        "Algorithms" => [
        		"JCSA"     => "JCSA.md",
                "DepthWMM" => "DepthWMM.md"
        ]
    ],
    modules = [RGBDSegmentation]
)
