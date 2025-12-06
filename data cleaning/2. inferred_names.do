/*
 inference_names.do

 Purpose:
 - Import the cleaned parliamentary speeches CSV.
 - Report counts specifically for missing `name`, missing `speech`, and missing both `party` and `party_family`
 - Try to infer missing `name` values by extracting a speaker name from the beginning of the `speech` text
 - Compute and print missing-value counts for every variable (treat blank strings as missing)
 - Save the filtered dataset`

 */

// -------------------------------
// Configuration: set your filenames
// -------------------------------

use "insert_dataset_with_proper_dates.dta", clear

// -------------------------------
// 0) Compute missing counts per variable
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
// 1) Try to infer missing `name' from start of the `speech' text
//    - Common speech headers: "MARIO ROSSI - ...", "Mario Rossi (PD): ...", or lines that start with the speaker name
//    - We'll attempt several regex patterns in order and collect inferred candidates
// -------------------------------
generate str60 inferred_name = ""
generate byte inferred_flag = 0

* Target rows where the printed 'speaker' is missing/generic
gen byte _target = inlist(upper(speaker),"","PRESIDENTE","PRESIDENZA","VICEPRESIDENTE","PRESIDENTE DELLA CAMERA")

if text != "" {
	* A) ALL-CAPS name at start, followed by comma/dash/colon
	replace inferred_name = ustrregexs(1) if _target & ///
		ustrregexm(text,"(?s)^\s*([A-ZÀ-Ü''\-\.]+(?:\s+[A-ZÀ-Ü''\-]+){1,3})\s*[.,]")

	* B) Capitalized name followed by (party)
	replace inferred_name = ustrregexs(1) if _target & missing(inferred_name) & ///
		ustrregexm(text,"(?s)^\s*([A-Z][A-Za-zÀ-ÿ''\-\.]+(?:\s+[A-Z][A-Za-zÀ-ÿ''\-\.]+){1,3})\s*\(")
}

replace inferred_name = ustrtrim(inferred_name)
replace inferred_flag = inferred_name!="" if _target

* (Optional) overwrite speaker when we inferred a name
replace speaker = inferred_name if _target & inferred_flag

* Report & show a few examples
drop if speaker==""

drop _target

save "names.dta", replace

// -------------------------------
// 2) Save filtered dataset to CSV
// -------------------------------
capture confirm file "$OUTPUT"
// export as CSV (comma delimited). Replace if exists.
export delimited using "names.csv", replace
di as txt "\nFiltered dataset saved to: $OUTPUT"

// End of script
