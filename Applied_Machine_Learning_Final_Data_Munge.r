# R file to munge data before EDA
library(tidyverse)
library(magrittr)
clinical_info <- read_tsv("~/Desktop/Academics/Applied_Machine_Learning/clinical.project-TCGA-KIRC.2019-04-18/clinical.tsv")

clinical_info %>%
  group_by(race) %>%
  count(primary_diagnosis) -> diagnoses_per_race

diagnoses_per_race

head(clinical_info)

ggplot(diagnoses_per_race, aes(x = primary_diagnosis), group = race) + geom_bar()

clinical_info %>%
  group_by(tumor_stage) %>%
  count() %>%
  print()

# The entry for this variable is the same for all cases. The variable can be dropped
clinical_info %>%
  group_by(tissue_or_organ_of_origin) %>%
  count()

clinical_info %>%
  select(-tissue_or_organ_of_origin)

## Remove these columns per shared analysis
names(clinical_info)
# remove the varibles that do not have any variation or
clinical_info %<>%
  select(-case_id, -project_id, -classification_of_tumor, -last_known_disease_status, -days_to_last_known_disease_status, -days_to_recurrence, -tumor_grade, -morphology, -tissue_or_organ_of_origin, -progression_or_recurrence, -site_of_resection_or_biopsy, -therapeutic_agents, -treatment_intent_type, -treatment_or_therapy, -days_to_birth, -prior_malignancy)
names(clinical_info)

# replace submitter_id with participant ID using gsub
# clinical_info[1:5,  1]
clinical_info$submitter_id <- gsub("TCGA-[A-Z0-9]{2}-", "", clinical_info$submitter_id)
# clinical_info[1:5,  1]

write_csv(clinical_info, "~/Desktop/Academics/Applied_Machine_Learning/clinical.project-TCGA-KIRC.2019-04-18/cleaned_clinical_R.csv")
