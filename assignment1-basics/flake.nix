{
  description = "Dev shell for Stanford CS336 Assignment 1 (basics) — uv-managed Python/PyTorch env";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # The assignment requires Python >=3.12,<3.14; 3.13 has torch 2.11 in nixpkgs.
        python = pkgs.python313;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            python
            pkgs.git
          ];

          env = {
            # Reuse the Nix Python instead of uv downloading its own build.
            UV_PYTHON = "${python}/bin/python3";
            UV_PYTHON_DOWNLOADS = "never";
            # PyTorch wheels are large; keep them in a stable location.
            UV_LINK_MODE = "copy";
          };

          shellHook = ''
            # uv-installed wheels (torch, numpy, ...) are built for standard
            # Linux and need libstdc++/zlib at runtime; make them loadable.
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            echo "CS336 assignment1-basics dev shell"
            echo "  uv run pytest      # run unit tests"
            echo "  uv run <file.py>   # run scripts"
            echo "First 'uv run' / 'uv sync' will resolve and download the"
            echo "Python packages (torch etc.) into .venv."
          '';
        };
      });
}
