library(readxl)
library(ggplot2)


### PLOTING TRENDS

data <- read_excel("data/data_for_plots.xlsx")

ggplot(data, aes(x = factor(draft_year), y = Three_attempts, fill = factor(draft_year))) +
  geom_boxplot() +
  labs(title = "Year vs 3 Point Attempts",
       x = "Draft year",
       y = "3 Point Attempts") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "none")

ggplot(data, aes(x = factor(draft_year), y = PTS, fill = factor(draft_year))) +
  geom_boxplot() +
  labs(title = "Year vs Points per Game",
       x = "Draft year",
       y = "PPG") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "none")
