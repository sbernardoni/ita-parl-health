/*
 date_cleaning.do

 Purpose:
 - Import the raw parliamentary speeches CSV (camera_2006_2022.csv by default).
 - Parse and normalize the date column (day-first - Italian dates)
 - Keep only observations from 2018-01-01 onward (2018-2022)
 - Compute and print missing-value counts for every variable (treat blank strings as missing)
 - Save the filtered dataset

*/

// -------------------------------
// Configuration: set your filenames
// -------------------------------
global INPUT "insert_your_input.csv"
global OUTPUT "insert_your_output.csv"

// -------------------------------
// 1) Check input exists and import
// -------------------------------
capture confirm file "$INPUT"
if _rc {
    di as error "Input file '$INPUT' not found in `c(pwd)'."
    di as error "Put the CSV in the working directory or update the $INPUT global in this .do file."
    exit 198
}

// Try import with auto-detection first; fall back to common delimiters
capture noisily import delimited using "$INPUT", clear
if _rc {
    di as txt "First import attempt failed; retrying with semicolon delimiter..."
    capture noisily import delimited using "$INPUT", delimiter(";") clear
    if _rc {
        di as txt "Second attempt failed; retrying with tab delimiter..."
        capture noisily import delimited using "$INPUT", delimiter("\t") clear
        if _rc {
            di as error "All import attempts failed. Edit this .do and try an explicit import command with the correct delimiter/encoding."
            exit 198
        }
    }
}

di as txt "Imported dataset: `c(filename)' -- `c(N_obs)' observations, `c(N_var)' variables"

// -------------------------------
// 2) Drop useless columns
// -------------------------------
drop v15-v320

// -------------------------------
// 3) STRING WHITESPACE - Normalize whitespace in string variables
//    - Trim leading/trailing spaces
//    - Keep empty strings as Stata missing for strings ("" is treated as missing)
// -------------------------------
ds, has(type string)
local strvars `r(varlist)'
foreach v of local strvars {
    // Trim whitespace
    replace `v' = trim(`v')
    // Replace strings that are only spaces with empty string
    replace `v' = "" if `v' == ""
}

// -------------------------------
// 4) DATE HANDLING
// -------------------------------

* Change "date" into usable variable "d"
gen double d = date(date, "DMY")
format d %td

* Drop all observations before 2018 or empty dates
drop if d <= date("31/12/2017", "DMY")
drop if date==""
drop if d==.

replace year=year(d)

// ------------------------------- [OPTIONAL for user to check]
// 5) Compute missing counts per variable
//    - For every variable show how many observations are missing
//    - Treat empty strings and missing numerics as missing (Stata treats empty string as missing)
// -------------------------------
di as txt "\nMissing counts per variable (variable : count_missing)"
foreach v of varlist _all {
    quietly count if missing(`v')
    local n = r(N)
    di as res "`v' : `n'"
}

// -------------------------------
// 6) Save filtered dataset to CSV
// -------------------------------
save "2018filtered.dta", replace
capture confirm file "$OUTPUT"
// export as CSV (comma delimited). Replace if exists.
export delimited using "2018filtered", replace
di as txt "\nFiltered dataset saved to: $OUTPUT"

// End of script
