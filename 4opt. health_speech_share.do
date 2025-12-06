
// import ONLY ONE of the following two datasets

* BERT-filtered dataset

import delimited "D:\OneDrive - Università Commerciale Luigi Bocconi\Desktop\Università\Foundations of Social Sciences\Group Project\doc_assignments_with_health_umberto_2.csv"

* LDA-filtered dataset

import delimited "D:\OneDrive - Università Commerciale Luigi Bocconi\Desktop\Università\Foundations of Social Sciences\Group Project\df_clean_total_lda_final.csv"

* Create dummy for period pre-covid (0) and post (1)

gen byte period = 0
replace period = 1 if quarter>"2020q2"

* Right only
tabulate is_health_intersection period if side3=="Right", col

* Left only
tabulate is_health_intersection period if side3=="Left",  col

drop if is_health_by_keywords=="False"
drop if is_health_by_semantic_umberto=="False"
drop if is_health_intersection=="False"
drop if is_health_union=="False"

gen str side = ""
replace side = "Left" if side3=="Left"
replace side = "Right" if side3=="Right"
replace side = "Other" if side3=="M5S"
replace side = "Other" if side3=="Center"



export delimited using "D:\OneDrive - Università Commerciale Luigi Bocconi\Desktop\Università\Foundations of Social Sciences\Group Project\2018filtered_notextoriginal", replace