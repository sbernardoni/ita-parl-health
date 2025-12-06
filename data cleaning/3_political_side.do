/*
 political_side.do

 Purpose:
 - Create new column side3 that assigns each observation to left, right, center or M5S
*/

// -------------------------------
// Configuration: set your filenames
// -------------------------------

use "insert_your_filtered_dataset_with_dates_and_names", clear

// -------------------------------
// 6) Compute missing counts per variable
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
// 8) Try to infer missing party_family from party_name
//    - Many families are known and present in the dataset but missing for some observations
// -------------------------------

use "names.dta", clear

// Remove sentences from the president
drop if party_name=="Chair"

// Remove sentences from people with "Others" as both party name and party family
drop if party_family=="Others" & party_name=="Others"

// By definition, all the members in "Gruppo Misto", here called "#NAME?", do not have a group of affiliation. We also drop the remaining "Others" (after inferring all possible values), whom affiliation cannot be inferred.
drop if party_name=="#NAME?"
drop if party_name=="Others"

* Normalize strings so they always match
replace party_name = subinstr(party_name,"'","'",.)
replace party_name = subinstr(party_name,"–","-",.)
replace party_name = ustrregexra(party_name,"\s+"," ")
replace party_name = strtrim(party_name)

* Make a crosswalk and merge
preserve
clear
input str60 party_name str10 side3 str15 side5
"#NAME?"                              "Left"            "Left"
"Associativo Italiani all'Estero"     "Center"          "Center"
"Centro Democratico"                  "Center"          "Center"
"Centrodestra"                        "Right"           "Center-right"
"Centrosinistra"                      "Left"            "Center-left"
"Chair"                               ""                "Other"
"Forza Italia - Il Popolo della Libertà" "Right"       "Center-right"
"Fratelli d'Italia"                   "Right"           "Right"
"Italia dei Valori"                   "Center"          "Center-left"
"L'Ulivo"                             "Left"            "Center-left"
"Lega"                                "Right"           "Right"
"Lega Nord"                           "Right"           "Right"
"MAIE"                                "Center"          "Center"
"Movimento 5 Stelle"                  "M5S"             "M5S"
"Others"                              ""                "Other"
"Partito Democratico"                 "Left"            "Center-left"
"Partito Popolare Italiano"           "Center"          "Center"
"Radicali"                            "Center"          "Center-left"
"SVP-PATT"                            "Center"          "Center"
"Scelta Civica"                       "Center"          "Center"
"Sinistra Ecologia Libertà"          "Left"            "Left"
"Südtiroler Volkspartei"             "Center"          "Center"
"USEI"                                "Center"          "Center"
"Union Valdôtaine"                   "Center"          "Center"
"Unione / Centro"                     "Center"          "Center"
end
tempfile xwalk
save `xwalk'
restore

merge m:1 party_name using `xwalk', keepusing(side3 side5) nogen

gen quarter=qofd(d)
format quarter %tq

save "completev2.dta", replace

drop side5
drop party_id_itaparl
drop party_id_parlgov
drop pageid_wiki
drop row_id
drop doc_id
drop chair

save "final_cleaned.dta", replace 

// -------------------------------
// 9) Save filtered dataset to CSV
// -------------------------------
capture confirm file "$OUTPUT"
// export as CSV (comma delimited). Replace if exists.
export delimited using "final_cleaned.csv", replace
di as txt "\nFiltered dataset saved to: $OUTPUT"

// End of script
