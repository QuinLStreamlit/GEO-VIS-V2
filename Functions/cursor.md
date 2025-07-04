# Cursor Rules for Geotechnical Data Analysis Project

This file contains cursor-specific rules and guidelines for working with the LGCFR geotechnical data analysis codebase.

## Core Development Principles

### 🚨 CRITICAL: Preserve Existing Functionality
- **NEVER** overwrite or break existing working functions
- **ALWAYS** maintain backward compatibility with existing function signatures
- **ONLY** add new features or improve existing capabilities
- When modifying functions, add new parameters with default values to preserve existing behavior
- Test that current functionality still works after any modifications

### Code Organization and Structure
- Follow the existing package structure with Functions/ as the core module
- Place plotting functions in separate files following the `plot_*.py` naming convention
- Keep data processing functions separate from visualization functions
- Maintain the comprehensive `QL_functions_overall.py` as the main function library

## Data Standards and Conventions

### Required Column Names
- Use `Hole_ID` for borehole identifiers
- Use `Material` for material classifications
- Use `From_mbgl` for depth measurements (meters below ground level)
- Follow geotechnical standards for material classifications
- Organize test results by test type with standardized naming

### Data Processing Rules
- Implement depth-based matching with appropriate tolerance
- Handle missing data gracefully with validation
- Ensure data integrity checks for depth intervals and borehole matching
- Convert measurements to consistent units (use `parse_to_mm()` for size conversions)

## Code Style and Best Practices

### Python Standards
- Use the scientific Python stack: pandas, numpy, matplotlib, seaborn, openpyxl, scipy
- Include comprehensive input validation and error handling
- Write functions that support extensive customization (colors, markers, layouts)
- Implement statistical analysis integration with visualizations

### Function Design
- Create functions with clear, descriptive names following geotechnical terminology
- Support faceting, stacking, and statistical overlays in plotting functions
- Include professional Excel formatting with color coding and styling
- Provide default parameters that work for common use cases

### Documentation
- Include docstrings that explain geotechnical context and expected data formats
- Document data structure requirements and validation rules
- Explain statistical methods and geotechnical standards being applied
- Provide examples of typical usage patterns

## File and Output Management

### Directory Structure
- Read raw data from Input/ directory organized by test type
- Save outputs to timestamped directories in Output/
- Generate both Excel reports and PNG visualizations
- Maintain separate files for different types of geotechnical analyses

### Output Standards
- Create professional Excel reports with formatted styling and color coding
- Generate publication-quality plots and charts
- Include metadata and timestamps in output files
- Ensure outputs follow geotechnical reporting standards

## Testing and Validation

### Data Validation
- Implement checks for required columns and data types
- Validate depth measurements and interval consistency
- Check for reasonable value ranges based on geotechnical norms
- Handle edge cases and missing data appropriately

### Function Testing
- Test functions with typical geotechnical datasets
- Verify backward compatibility when modifying existing functions
- Ensure new features don't break existing workflows
- Validate statistical calculations against known results

## Geotechnical Domain Knowledge

### Key Test Types
- Atterberg limits and plasticity classifications
- CBR (California Bearing Ratio) and swell tests
- UCS (Unconfined Compressive Strength) and IS50 point load tests
- Particle size distribution (PSD) analysis
- Triaxial and consolidation tests
- Emerson class and other soil classification tests

### Analysis Focus
- Soil plasticity and classification
- Rock strength correlations and relationships
- Statistical analysis of geotechnical parameters
- Professional visualization for engineering reports
- Integration with geological modeling outputs (Leapfrog)

## AI Assistant Guidelines

### When Working with This Codebase
- Prioritize understanding the geotechnical context before making changes
- Ask clarifying questions about test standards or data interpretation when uncertain
- Suggest improvements that enhance functionality without breaking existing code
- Focus on creating robust, professional-quality outputs suitable for engineering reports
- Consider the specific needs of geotechnical engineers and engineering workflows

### Code Modification Approach
- Always review existing function signatures before proposing changes
- Suggest additive improvements rather than replacements
- Maintain consistency with existing coding patterns and naming conventions
- Ensure any new features integrate well with the existing scientific Python workflow 