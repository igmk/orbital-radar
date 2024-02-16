# Orbital-Radar

This repository contains the source files of the Orbital-Radar documentation website generated with Sphinx.

## Updating the Documentation

The input required for Sphinx to generate these source files are located in the main branch under `/doc`. To update the documentation website, follow these steps:

1. Navigate to the main branch:
   `cd doc`

2. Modify the `.rst` files to change the documentation content.

3. Generate the HTML documentation: `make html`

4. Copy the generated files from `doc/_build/html` into the root directory of the `gh-pages` branch.

5. Push your changes to the `gh-pages` branch.

Ensure that the `README.md` file and `.nojekyll` remain unchanged. All other files can be replaced.
