library(sf)
library(tensorflow)
library(keras)
library(tibble)
library(readr)
library(tidyverse)

#2 Variables (Pop Size, Lag Pop Size), W5

#Dataset is a Spatial Panel, consisting of 65445 rows representing individual European municipalities, and 42 columns representing annual population count and spatial lag of population count time series. 
setwd("C:/Users/niall/Documents/PhD/Paper 3/Data/Country.Files/CSV")
Data <- read.csv("All.Countries.csv")
Data <- Data[-c(1,2,26,47)]
Data <- na.omit(Data)

Data <- Data %>%
  filter(if_all(starts_with("X2"), ~ . >1))
filter(if_all(starts_with("X2"), ~ . <18234))

Pop.Change <- Data %>%
  dplyr::select(Area, starts_with("X")) %>% 
  pivot_longer(cols = starts_with('X'), names_to = "Year", values_to = "Population") %>% 
  mutate(Year = as.numeric(sub("X", "", Year)))

Lag.Pop.Change <-  Data %>%
  dplyr::select(Area, starts_with("lag_")) %>% 
  pivot_longer(cols = starts_with('lag'), names_to = "Year", values_to = "Lag Population") %>% 
  mutate(Year = as.numeric(sub("lag_change_", "", Year)))

# Merge the 'change' and 'lag_change' long format data frames
Data <- cbind(Pop.Change, Lag.Pop.Change) 
Data <- Data[-c(4:5)]
rm(Pop.Change, Lag.Pop.Change)

data <- as.matrix(Data[, -c(1, 2)])

# Set the column names of the data matrix
colnames(data) <- c("Population", "PopulationLag")

num_areas <- 65058
num_variables <- 2

# Add row names for identification
row_names <- rep(paste("Area", 1:num_areas), each = 20)
rownames(data) <- row_names

# Normalize the data (optional but recommended for LSTM)
index <- rep(rep(c(1, 2), c(14, 6)), length.out = nrow(data))
train <- data[index ==1, , drop = FALSE]
mean <- apply(train, 2, mean)
std <- apply(train, 2, sd)
normalized_data <- scale(data, center = mean, scale = std)
rm(train, Data)
#print(normalized_data)

# Create sliding window sequences
window_size <- 5
# Create empty lists to store sequences and targets for each area
sequences <- vector("list", num_areas)
targets <- vector("list", num_areas)

# Generate sequences and targets for each area
for (i in 1:num_areas) {
  # Calculate the starting row index for the current area
  start_row <- (i - 1) * 20 + 1
  
  # Extract the population data for the current area
  area_data <- normalized_data[start_row:(start_row + 20 -1), ]
  
  # Create empty matrices to store sequences and targets for the current area
  area_sequences <- matrix(0, nrow = nrow(area_data) - window_size, ncol = window_size * num_variables)
  area_targets <- matrix(0, nrow = nrow(area_data) - window_size, ncol = 1)
  
  
  # Generate sequences and targets for the current area
  for (j in 1:(nrow(area_data) - window_size)) {
    # Extract the sliding window sequence from the area data
    area_sequences[j, ] <- as.vector(t(area_data[j:(j + window_size - 1), c("Population", "PopulationLag")]))
    
    # Extract the target value for the next time step
    area_targets[j, ] <- area_data[j + window_size, "Population"]
  }
  
  # Store the sequences and targets in the respective lists
  sequences[[i]] <- area_sequences
  targets[[i]] <- area_targets
}


# Combine sequences and targets for all areas
all_sequences <- do.call(rbind, sequences)
all_targets <- do.call(rbind, targets)

#this results in a stacked matrix, where each time step in the sequence contains the values of both variables
#i.e. Sequence 1: [Population_Change_t, Lag Population Change_t, Population_Change_t+1, Lag Population Change_t+1, Population_Change_t+2, Lag Population Change_t+2]

# Create an index vector to specify the pattern of training and test sets
index <- rep(rep(c(1, 2), c(14, 1)), length.out = nrow(sequences))

# Split the data into training and test sets based on the index
train_data <- all_sequences[index == 1, , drop = FALSE]
train_targets <- all_targets[index == 1, , drop = FALSE]
test_data <- all_sequences[index == 2, , drop = FALSE]
test_targets <- all_targets[index == 2, , drop = FALSE]

# Reshape the data to match LSTM input shape
train_data <- array_reshape(train_data, c(dim(train_data)[1], window_size, num_variables))
test_data <- array_reshape(test_data, c(dim(test_data)[1], window_size, num_variables))
#print(train_data)
#print(test_data)

# Build the LSTM model
model <- keras_model_sequential()
model %>%
  layer_lstm(units = 48, input_shape = c(window_size, num_variables), activation = "relu") %>% 
  layer_dense(units = 1)

# Compile the model
model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(learning_rate = 0.001)
)

# Train the model
epochs <- 8
batch_size <- 14
model %>% fit(
  train_data, train_targets,
  epochs = epochs,
  batch_size = batch_size,
  validation_data = list(test_data, test_targets)
)

# Make predictions
predicted_targets <- model %>% predict(test_data)

# Denormalize the predictions
predicted_targets <- predicted_targets * std[1] + mean[1]

Data <- read.csv("All.Countries.csv")
Data <- na.omit(Data)
Data <- Data[-c(26,47)]

Accuracy <- read.csv("All.Countries.csv")
Accuracy <- na.omit(Accuracy)
#GID and X2020
Accuracy <- Accuracy[c(5,26)]
Accuracy <- merge(Data, Accuracy, by.x = "GID", by.y = "GID", sort = FALSE)
Accuracy <- Accuracy[c(46)]

Accuracy <- cbind(Accuracy,predicted_targets)
colnames(Accuracy) <- c("X2020", "LSTM.V.W3")
Accuracy$Diff <- Accuracy$LSTM.V.W3 - Accuracy$X2020
Accuracy$MAPE <- (Accuracy$Diff / Accuracy$X2020) * 100
Accuracy$MAPE <- abs(Accuracy$MAPE)
mean(Accuracy$MAPE)
median(Accuracy$MAPE)