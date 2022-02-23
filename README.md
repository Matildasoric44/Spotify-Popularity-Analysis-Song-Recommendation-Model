# Spotify-Popularity-Analysis-Song-Recommendation-Model
## Introduction
In this notebook we looked at Spotify's data and analyzed popularity of songs released from 1921 to 2020. We also built a model which recommends 10 similar songs based on one song's attributes.
## Importing models and data
We started off by importing module functions and all dependencies we need. We then uploaded our data using `pandas` and looked at the top 3 rows to get a quick overview of what our data looks like. We then used pandas' built-in `.info()` function to learn more about our data. We found that almost all of our 19 columns are either integers or floats. Four columns: `artists`,`id`,`name`, and `release_date` have a type of a object which we will change later in the notebook. The dataset contains 170,653 rows and we can also see that there is no missing values.
## Data Features Explanations
`valence`:A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).\
`year`: Year when the song was released\
`acousticness`: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.\
`artists`: Artists featured in the song\
`danceability`:Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.\
`duration_ms`:The duration of the track in milliseconds.\
`energy`:Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.\
`explicit`: Has the value of 1 if a song is exlplicit, 0 if it is not.\
`id`:The Spotify ID for the track.\
`instrumentalness`:Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.\
`key`:The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.\
`liveness`:Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.\
`loudness`:The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.\
`mode`:Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.\
`name`: Title of the song
`popularity`: The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are.\
`release_date`: The date song was released\
`speechiness`:Speechiness detects the presence of spoken words in a track.\
`tempo`:The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
## Cleaning the Data
After taking a closer look at the release_date column we noticed that some dates only had a year and some had year-month-day format. We split the column into year, month, and day and filled missing values with NaN to get an insight into how many of those valuse were missing and if there is anything else we could do with the data we had.
We found that around 30% of the rows were missing day and month of the release. Since it was a such a high number we decided to not drop those rows but instead focus on other attributes that could play a significant role for our model.
## Exploratory Data Analysis
In order to see which features and how correlated they are we made a correlation matrix between all columns. Then we wanted to focus just on `popularity` so we plotted a correlation matrix of `popularity` with all the columns. We noticed high positive correlation between `popularity` and `year` with a value of 0.86. It is important to mention `loudness` had a positive correlation of 0.46, and `energy` had a positive correlation of 0.59.
`acousticness` had a negative correlation to `popularity` of - 0.57.
## `Popularity` Distribution
Since our entire project is based on `popularity` we decided to look at the distribution of its values and how they have changed over the years. The histogram showed taht the `popularity` scores were unbalnced and that 16% of the scores had a value of 0 while only 9% of the scores had a values above 60.  We also plotted a line graph that represented an anverage popularity score over the years. In the recent years the popularity scores tend to get a higher value than in the 1920's, 30's and 40's. 
## Song Recommendation Model 
Our goal was to make a model that would recommend 10 songs based on a listener's taste in music. We used `sklear.neighbors` and its NearestNeighbors to fit the model with X being `valence`,`year`,`acousticness`, `danceability`, `energy`, `explicit`,`liveness`,`loudness`,`mode`,`speechiness`and y being `name`. We then used `kneighbors` and row's index to find 10 nearest neighbors and printed out those songs and their artists.
## Linear Regression Model 
We made a Linear Regression model to predict `popularity`.\
Firstly, we encoded columns with non-numerical data which were `artists` and `name` in order to use them in our model.Then we trained our model based on `popularity` > 60,  60 <=`popularity`< 0, and `popularity`<= 0. We sampled them based on the smallest value count out of the three subsets mentioned above. Concatenating all 3 subsets we made a balanced dataframe. Using this new data frame, we ran the score model to predict the best possible score. Moreover, after receiving score, recreated a linear model based on the new data framed.\
In order to get a combination of columns that would give us the best score, we made a function `powerset` that tries out all possible combinations and their scores and gives us the best combination of columns and the score our Linear Regression model would have. 
After running it we found that the highest score we could get was 0.8195 and it would be with these columns: `valence`,
 `year`,
 `acousticness`,
 `danceability`,
 `energy`,
 `explicit`,
 `liveness`,
 `loudness`,
 `mode`,
 `speechiness`.
 ###Creating Linear Regression Model Based on Best Columns Using Trained Dataframe
 We defined lr as a LinearRegression object and split our X and y variables. We used the **balanced** dataframe for our model.After fitting the model and making predictions our model got a score of 0.82. The score is decent but we wanted to look at our predictions and the actual values to see how good our model actually is. Plotting these values will also help us notice any outliers or clusters in the data.
Mean Absolute Error was 8.7 which means our predicions were missed by 8 points on average. This can mean 8 points too high or 8 points too low on our popularity scores prediction. Another observation was that our model was not not predicting any values above 80. We will look more into that in the next part.
###Analysing Why the Model Will Not Predict Values Above 80 While Showing Actual Values up to 100
We wanted to find out why our model would not predict values above 80 so we made a Linear Regression Model on `popularity` values higher than 60 and fit the model. Mean Absolute Error was 4.4 and R^2 was 0.05. but the model still didn't predict any values higher than 80.
# Findings & Suggestions 
After looking at our data, cleaning, analysing and making models we found that newer songs tend to be more popular. A few features that help a song become more popular are: more energy and loudness, and less acousticness.
We would suggest adding more newer songs to the dataset to balance it out with the older songs which are less popular. Since Spotify was founded in 2006 it is reasonable that songs after 2006 have more play counts that the ones made in the early 1920's. 
