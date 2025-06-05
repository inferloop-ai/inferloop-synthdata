@echo off
REM Create the main project directory
mkdir inferloop-synthetic

REM Create subdirectories and files
mkdir inferloop-synthetic\sdk
mkdir inferloop-synthetic\cli
mkdir inferloop-synthetic\api
mkdir inferloop-synthetic\examples
mkdir inferloop-synthetic\data\sample_templates
mkdir inferloop-synthetic\tests

REM Create empty files
type nul > inferloop-synthetic\sdk\__init__.py
type nul > inferloop-synthetic\sdk\base.py
type nul > inferloop-synthetic\sdk\sdv_generator.py
type nul > inferloop-synthetic\sdk\ctgan_generator.py
type nul > inferloop-synthetic\sdk\ydata_generator.py
type nul > inferloop-synthetic\sdk\validator.py
type nul > inferloop-synthetic\cli\main.py
type nul > inferloop-synthetic\api\app.py
type nul > inferloop-synthetic\api\routes.py
type nul > inferloop-synthetic\examples\notebook.ipynb
type nul > inferloop-synthetic\tests\test_sdk.py
type nul > inferloop-synthetic\pyproject.toml
type nul > inferloop-synthetic\README.md

echo Project structure created successfully!
