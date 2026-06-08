{
  description = "samplemaker-sparrow lithographic mask design package";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python313;
        pyPkgs = python.pkgs;

        samplemaker = pyPkgs.buildPythonPackage {
          pname = "samplemaker-sparrow";
          version = "5.4.8";
          pyproject = true;

          src = ./.;

          build-system = [
            pyPkgs.scikit-build-core
            pyPkgs.pybind11
          ];

          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
          ];

          buildInputs = [
            pkgs.boost
          ];

          dependencies = [
            pyPkgs.matplotlib
            pyPkgs.numpy
          ];

          # scikit-build-core invokes cmake internally; skip Nix's cmake configure hook
          dontUseCmakeConfigure = true;

          meta = with pkgs.lib; {
            description = "Lithographic mask design package";
            homepage = "https://github.com/SparrowQuantum/samplemaker";
            license = licenses.bsd3;
          };
        };
      in
      {
        packages = {
          default = samplemaker;
          samplemaker = samplemaker;
        };

        devShells = {
          default = pkgs.mkShell {
            packages = [
              pkgs.ruff
              python.withPackages (ps: [
                samplemaker
                ps.pytest
              ])
            ];
          };
          uv = pkgs.mkShell {
            packages = with pkgs; [
              uv
              cmake
              ninja
              boost
            ];
          };
        };
      }
    );
}
