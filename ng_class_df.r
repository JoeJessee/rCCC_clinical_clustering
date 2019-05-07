ngs class df
library(tidyverse)
gene_name <- c('gene_1', 'gene_2', 'gene_3', 'gene_4', 'gene_5', 'gene_6', 'gene_7', 'gene_8', 'gene_9', 'gene_10')
drug_rep1 <- c(9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 11.1, 11.2, 11.3, 11.4)
drug_rep2 <- c(10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 12.1, 12.2, 12.3, 12.4)
control_rep1 <- c(1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 3.1, 3.2, 3.3, 3.4)
control_rep2 <- c(2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 4.1, 4.2, 4.3, 4.4)

gene_count_matrix = data_frame(gene_name, drug_rep1, drug_rep2, control_rep1, control_rep2)
