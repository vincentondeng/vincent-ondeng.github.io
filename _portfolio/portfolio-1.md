---
title: "Prediction of Early Marriages: A data-driven approach to solving a real challenge"
excerpt: "Early Marriage remains a critical issue in Country X, with profound social, economic and health consequences. In this project, I am going to predict explore some factors that are likely to contribute to Early Marriages in this country. Additionally, I am going to anyalyze how using Apache Spark instead of sklearn reduces model training time significantly. <br/><img src='/images/500x300.png'>"
collection: portfolio
---

### The Task

The Ministry of Health in the country has expressed concern about the prevalence of early marriages among young individuals (both men and women). They have tasked you with investigating the factors contributing to early marriages. For the purpose of this analysis, individuals who get married at the age of 18 or younger are classified as having married early or belonging to the early marriage category. 
Beyond conducting exploratory analysis, the Ministry has requested that you develop a model to predict whether a person is likely to marry young, based on factors such as place of residence, household size, parents' education levels, and other relevant variables. In summary, these are the project goals.

### Objectives
- Perfom explotaory analysis to understand early marriages
- Build a Machine Learning model which can predict whether a person will get married early or not.
- Report on the model performance and efficacy 

A detailed dexcription of the code and further explanation can be found in the following notebook: 

### Dataset Description
The data is from a population and housing census of some country X not identified for privacy reasons although this data is a very small subset of the actual data. Each row in the data represent a single individual in the population. A summary of column description is provided below:

- **Geographic identifiers**: PROVINCE, REGION, DISTRICT, COMMUNE, MILIEU
```
 IDMEN, IDINDIV. This type of data has a somewhat hierarchical structure. We have a household (think of it as family), IDMEN-household ID. Within each household, we have individuals, IDINDIV - individual
 > 
```
- **MILIEU**: A classification of whether this person lived in urban or rural area. 2-Rural, 1-Urban

- **Sex**. P05==>[1 - Male 2 - Female]

- **P19 Languages spoken**. What languages the person can speak.This variable is split into 4 variables as follows: P19MG, P19FR, P19AN, P19AU for local language, English, French and any other language.

- **P20, Literacy**. Whether the person can read and write any of the 3 languages given. Note that there three variables each representing each language. A local language, French and English. For each language, the value 1 means they can read and write in the language while 2 means they cannot.The variables are P20MG (local language), P20FR (French), P20AN (English), P20AU (other).

- **P03**: whether the person is the head of the household, wife. child etc==>[0- Chef de Ménage (CM) 1- Conjoint(e) (CJ) 2- Fils/Fille3- Père/Mère 4- Beau-Père/Belle-Mère 5- Beau-Fils/Belle-Fille 6- Petit fils/Petite-fille Autre Proche du CM 8- Autre proche du CJ 9 -Sans lien de parenté]

- **Age**:. Person's date of birth is given by column P07M (month of birth), P07A (year of birth) and P08 (age)

- **Marital status**: P28 (whether the person is married or not)==>[1- Célibataire, 2- Marié(e), 3- Divorcé(e)/Séparé(e), 4- Veuf(ve)]. This question is asked to residents who are 12 years or older.

- **Age at first marriage**. P29 (age at marriage).The question was like this: How old was when he/she got married for the first time?

- **School attendance**: P21 ==>[0 N'a Jamais fréquenté 1-A fréquenté 2- Fréquente actuellement]

- **Highest school level attended**:P22N. This variable represents highest level of school attended. The question was asked like this: What is the highest level of education that (name) achieved during his studies? Preschool; 2. Primary-school; 3. Secondary; 4. Technical college; 5. University

- **Number of years of school completed at a particular level**: P22C Years completed at that level. A value of 0 means the person didnt complete the first year of education at that level. Preschool(0-2); Primary-school(0-5);Secondary(0-7); Technical college (0-7); University (0-7)

- **Whether the person worked or not**: P23==> [1- 0ccupé 2- Chômeur 3- En quête du 1er emploi 4- Ménagère 5- Elève/Etudiant 6- Retraité 7- lncapacité à travailler 8- Autre]

### Task 1 - Loading and Subsetting the Data
To efficiently load and work with data, I began by setting up a Spark session using 8 cores to facltate faster operations. I allocated 4GB of memory for both the driver and executor, and enabled off-heap memory to handle larger datasets effectively. The number of shuffle partitions was set to 40, balancing parallelism with performance. I loaded the dataset using Spark’s read.csv() function, ensuring that Spark automatically inferred the schema and processed the CSV file with a comma delimiter. To optimize processing, I filtered the data down to just the columns I needed using the select() function.

To enhance processing efficiency, I repartitioned the DataFrame into 10 partitions, ensuring the workload was distributed well across available resources. After processing, I coalesced the DataFrame into a single partition to minimize the number of output files when writing the data back to disk.
```python
spark = SparkSession.builder\
                .appName("LargeDatasetProcessing")\
                .master(f"local[8]")\
                .config("spark.driver.memory", "4g")\
                .config("spark.executor.memory", "4g")\
                .config("spark.memory.offHeap.enabled", "true")\
                .config("spark.sql.shuffle.partitions", "40")\
                .config("spark.memory.offHeap.size", "1g")\
                .config("spark.driver.bindAddress", "127.0.0.1") \
                .getOrCreate()
```

Finally, I wrote the processed data to a new CSV file, using the overwrite mode to replace any previous output, ensuring that the final dataset was ready for preprocessing.

This approach allowed me to balance memory usage, parallel processing, and disk I/O effectively, ensuring smooth handling of the large csv while keeping perfomance optimal.

### Task 2: Data Preprocessing
The goal of this task is to transform the raw data into a clean, structured and feature rich dataset that can be effectively used to build a machine learning model. In this section, I focused on improving the data quality, handling missing values, and generating additional variables that will be crucial for building the model.

Machine learning models require categorical variables to be encoded as numeric values. I will apply the following techniques; For nominal categorical variables without an inherent order (e.g., gender, region), I will apply one-hot encoding to create binary features for each category, making the data suitable for most machine learning algorithms.

Feature engineering is crucial to enhance the model's predictive power. During this step, I will create new features that capture important relationships within the data: Interaction features: I will combine existing features that may have a meaningful relationship. For example, I might create a new variable such as "age x education level" to capture the interaction between these two variables.

### Add New Variables We Need
In some cases, key information we need to explore may not be readily available in the dataset. For example, to analyze households, we might need to create a new column to represent household size.

#### Household Size
Household size refers to the number of people in a household. The dataset provides a household identifier (`hh_id`) and an individual identifier (`indiv_id`). Using these, we can generate a new column called household_size.

#### Feature Engineering: Creating Additional Variables
Feature engineering is the process of transforming raw data into meaningful features that improve the performance of machine learning models. This involves selecting, creating, modifying, or aggregating data attributes to make them more informative and relevant to the task at hand. Feature engineering is inherently a creative task—there are no strict rules. As a data scientist or machine learning practitioner, it's up to you to explore the data, consult domain experts, and study relevant literature to design and test new features.

For this analysis, we can consider creating the following features, which may influence the age at first marriage:

- Number of dependent children in the household: Defined as the number of individuals aged 15 and younger.
- Number of dependent adults in the household: Defined as the number of individuals aged 65 and older.

#### Household Level Variables
Note that we have two levels of analysis units here: the individual and the household. As such, variables such as household size, number of children, number of the eldery are all household level variables. Since the head of the family or head of the household has more power in determing what happens in the house, we can also add household head variables. Concretely, for each household, we can have variables named like this: `hoh_age`, `hoh_educ`, `hoh_literacy` etc.

### Task 3 - Explolatory Data Analysis (EDA)

Before any ML task, its important to understand the data. This is done by exploring the data to understand the data types, missing values, and the distribution of the data. This is important as it helps in understanding the data and the features that can be used in the ML model.

Major Variables explored are `age_at_marriage` and `level_of_education`
First, I explored age at first marriage, as it directly reflects the concept of early marriage. By examining the distribution of this variable, I aimed to identify common age ranges at which individuals tend to marry, and whether certain patterns emerge, such as a higher frequency of early marriage within specific age groups. Additionally, I looked for potential outliers or any skew in the data that might indicate extreme cases of early marriage.

Next, I focused on highest level of education, as education is often a significant predictor of marriage timing. I analyzed the relationship between education levels (e.g., primary, secondary, university) and age at first marriage. This helped in identifying whether individuals with lower levels of education tend to marry earlier, while those with higher education levels might delay marriage. The interaction between education and early marriage could provide insights into socio-economic factors that influence marital decisions.

![alt text](image-2.png)

The age at first marriage is skewed to the right with many individuals getting married about 20 years.A right-skewed histogram of age at first marriage indicates that most individuals are marrying at younger ages, but there is a long tail extending towards older ages.

```
Summary of Age at First Marriage by Highest Education Level
+-------------------+--------------------+--------+---------+
| highest_education |        mean        | median |  count  |
+-------------------+--------------------+--------+---------+
|     Preschool     | 20.23346382767661  |  19.0  | 112904  |
|  Primary-school   |  20.4047519741933  |  20.0  | 4457137 |
|     Secondary     | 21.609568584169892 |  20.0  | 2790695 |
| Technical college | 23.140605523160968 |  22.0  | 119533  |
|    University     | 25.30795173033027  |  25.0  | 369756  |
+-------------------+--------------------+--------+---------+
```
The table above shows a clear trend between education level and the age at first marriage. Individuals with only preschool education marry the earliest, with an average age of around 20.2 years. As education increases, the age at first marriage also rises. Those with primary-school education marry at an average age of 20.4, while individuals with secondary education marry slightly later, at 21.6 years on average.

For those with technical college education, the average age increases further to 23.1 years, and university graduates have the latest average age at first marriage at 25.3 years. This trend suggests that higher levels of education are associated with a delay in marriage, likely due to factors like career and financial stability. In contrast, lower education levels are linked to earlier marriages.

### Task-4: Prepare data for ML model building

To predict early marriage, the target variable is created from age at first marriage. A threshold age, such as 20 years, is used to define early marriage. If someone marries at or before this age, the target variable is assigned a value of 1 (early marriage). If they marry after this age, the target variable is assigned a value of 0 (late marriage). This converts the age at first marriage into a binary classification variable.

In order to gain an initial understanding of which variables might be important for classification, I used a Random Forest model with 10 trees. This helped me get a rough idea of the variables that could potentially be useful in predicting early marriage, even though the results were not highly accurate. The Random Forest model provided an overview of the feature importances, giving me insight into which variables were likely contributing the most to the classification task. While this approach didn't provide precise feature rankings, it served as a useful starting point to identify the key features that might be worth further exploration and refinement in the modeling process.
```
Top 20 Most Important Features:
                      Feature  Importance
7             age_at_marriage        0.84
6                         age        0.05
54                      sex_1        0.02
21           head_household_1        0.02
3                     COMMUNE        0.01
4                       hh_id        0.01
2                    DISTRICT        0.01
38  local_language_literacy_0        0.00
39  local_language_literacy_1        0.00
5                    indiv_id        0.00
9                num_children        0.00
8                     hh_size        0.00
35        school_attendance_1        0.00
0                    PROVINCE        0.00
11                   MILIEU_1        0.00
41          french_literacy_1        0.00
16           working_status_3        0.00
31        highest_education_1        0.00
48                   french_0        0.00
50                  english_0        0.00
```

As for communes, their inclusion depends on their significance. If communes represent smaller geographic units with distinct socio-cultural characteristics that influence marriage timing, they could provide valuable insights. However, if the number of communes is large or their information overlaps with districts or regions, it might introduce redundancy.

### Task 5 - Model Building
Now I proceed to the model building section, where I implement different machine learning models and then compare their accuracy and perfomance in predicting Early Marriages of the dataset. These are Logistic regression, Random Forest and Gradient Boosting.

```python
def evaluate_models_sklearn(df_important_features):
    # Define the target variable and features
    target_var = 'early_marriage'
    features = df_important_features.columns[df_important_features.columns != target_var]

    # Split the data into training and testing sets
    X = df_important_features[features]
    y = df_important_features[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the models
    models = {
        "Logistic Regression": LR(max_iter=100, random_state=42,solver='saga'),
        "Random Forest": RF(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GBM(n_estimators=100, random_state=42),
        "Extra Trees": ETC(n_estimators=100, random_state=42, n_jobs=-1)
    }

    # Train and evaluate each model
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append([name, accuracy])

    # Print the results in a tabular format
    print(tabulate(results, headers=["Model", "Accuracy"], tablefmt="pretty"))
```
The results from the models indicate solid performance, with Gradient Boosting achieving the highest accuracy at 81.53%, followed closely by Decision Tree at 80.34% and Random Forest at 80.14%. These ensemble models perform better than Logistic Regression, which had the lowest accuracy at 79.89%. This suggests that models like Gradient Boosting and Random Forest, which can handle complex patterns and non-linear relationships, are more effective for predicting early marriage compared to simpler models like Logistic Regression.
```
+---------------------+--------------------+
|        Model        |      Accuracy      |
+---------------------+--------------------+
| Logistic Regression | 0.7793856191158197 |
|    Random Forest    | 0.7930157638786258 |
|  Gradient Boosting  | 0.815785951734737  |
|     Extra Trees     | 0.7938609482937423 |
+---------------------+--------------------+
```
The exact methodology is again implemented with spark. While the perfomance of the models remain relatively the same. Spark has a higher capability of handling large datasets once it has been cached.

The complete notebook can be accessed by following the following link: https://github.com/vincentondeng/BDA-with-Python/blob/main/vincent_ondeng_BDAPPr%20(1).ipynb
