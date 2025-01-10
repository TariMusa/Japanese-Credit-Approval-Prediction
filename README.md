# Introduction

The objective of this project is to predict credit card approval outcomes using the k-nearest neighbors (KNN) classification method. I would like to determine the most effective configuration of KNN to achieve the highest classification rate in predicting whether a credit card application is approved or declined by a bank. The project will involve analyzing various applicant features such as age, income, and credit history. I will also evaluate how some of these predictors relate to the likelihood of approval. Specific questions will include exploring which age group is more likely to be approved. By understanding the relationships between applicant characteristics and approval outcomes, the project aims to provide valuable insights into credit risk assessment, assuming that the approval of credit cards is based on some form of credit risk assessment and use those insights to create an effective prediction model.

The data used for this project is adapted from Kaggle:
[Credit Card Approval Dataset Kaggle](https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data)

The data from Kaggle is a cleaned version of the original dataset available on the UC Irvine machine learning repository:
[Credit Approval Dataset UCIrvine](https://archive.ics.uci.edu/dataset/27/credit+approval)

The dataset originates from a credit-issuing firm in Japan, which remains anonymous to ensure privacy and data protection. It is likely from 1992 or earlier. Notably, Japanese banks and their affiliates were largely excluded from credit card lending until 1992, as the credit market was dominated by non-bank lenders, such as shimpan kaisha, which specialized in consumer finance (Mann, 2001). This suggests that the institution providing this data was likely a non-bank entity. Additionally, the dataset reflects an emerging market rather than a developed one and should not be assumed to represent steady-state credit card approval trends in Japan.  It has 16 columns:

-   `Gender` categorical, encoded as 0 for female and 1 for male.

-   `Age` numerical, in years which is accurate to the nearest hundredth.

-   `Debt` numerical, the amount of outstanding debt the subject has. This is scaled ranging from 0 to 28.

-   `Married` categorical, encoded as 0 for no and this implies single, divorced etc and 1 for married.

-   `BankCustomer` categorical, an indicator for whether the applicant is a customer of the bank or not encoded as a 1 for yes and 0 for no.

-   `Industry` categorical, the industry the customer works with 14 different levels.

-   `Ethnicity` categorical ethnicity with levels Black, White, Latino, Asian and Other.

-   `YearsEmployed` the number of years the applicant has been employed for scaled ranging from 1 to 28.5.

-   `PriorDefault` a categorical indicator about whether the applicant ever previously defaulted on loans or any other forms of credit encoded as 1 for yes and 0 for no.

-   `Employment` a categorical indicator for the employment status of the applicant where 1 is for employed and 0 is for unemployed.

-   `CreditScore` a scaled numeric indicator that shows the applicant's credit score scaled and ranging from 0 to 67.

-   `DriversLicense` a categorical indicator for whether the applicant has a drivers' license or not where 1 is yes and 0 is no.

-   `Citizen` a categorical indicator for how the applicant is a citizen of the country. Levels are ByBirth, ByOtherMeans and Temporary.

-   `Zipcode` a scaled categorical indicator for the applicant's zipcode.

-   `Income` a scaled numerical indicator of the applicant's income rangning from 0 to 100000. Even after scaling, this remains the indicator with the largest range.

-   `Approved` the target variable, a categorical variable showing whether an applicant's application was accepted or not. Level 0 means the applicant was not approved and 1 means they were not approved.

To explore the use of KNN after encoding categorical variables, I transformed several features: `Ethnicity`, `Citizen`, and `ZipCode` into numeric variables, while keeping `Approved` as a factor. Although I considered encoding Industry, doing so would have significantly increased the model's complexity. Excluding it allowed me to achieve a very satisfactory classification rate.

For `ZipCode`, I first classified them into categories based on approval rates: high, medium, and low. This approach aimed to capture trends associated with areas that typically house wealthier individuals. While I didn’t have direct information about what each ZipCode represents, encoding them this way helped incorporate some of this contextual information into the model.The encoding for `Citizen` and `Ethnicity` were however standard.


# Data exploration

### Age 
<img width="537" alt="Screenshot 2025-01-10 at 1 29 10 PM" src="https://github.com/user-attachments/assets/3f65f3eb-8941-4386-950e-d2ee2fc2364a" />

This density plot examines the trends in credit card approval proportions across age groups. Both the approval and rejection distributions are right-skewed, indicating that credit card applicants are concentrated in the under 30 age group. In the context of 1992 Japan, this trend could reflect the economic activity of young adults entering the workforce. Alternatively, it might signal a broader cultural shift, with younger generations beginning to transition from Japan’s traditional cash-based economy to embracing credit.
The plot reveals that while individuals in their mid-20s receive the most approvals, they also face the highest number of rejections, suggesting this is the age group with the highest volume of credit card applications. Notably, rejection rates decline sharply after the mid-20s peak, indicating that stronger candidates for credit card approval tend to be older. For applicants under approximately 38 years old, rejections are more common than approvals. However, between the ages of 38 and 78, approval rates surpass rejection rates, suggesting that creditworthiness increases with age within this range. For this visualization, I used proportions insteaad of absolute counts so that I could provide provide insights on relative appproval rates for each age group and not just a reflection on the volume of applicants who get a certain outcome per each age group.


### Marital Status
<img width="539" alt="Screenshot 2025-01-10 at 1 30 52 PM" src="https://github.com/user-attachments/assets/ac642bb1-0cce-4c13-890b-73b44bccc619" />

This plot explores the relationship between marital status and credit card approval outcomes. The proportion of approved applicants who report being married is significantly higher than that of unmarried applicants. However, the majority of married applicants are not approved for credit cards, indicating that while married individuals are more represented in the approved group, they also form the majority in the rejected group. This suggests that most applicants for credit cards are married.

Among unmarried applicants, a larger proportion are not approved, which could reflect age-related factors. For instance, younger individuals, who are less likely to be married, may also be less likely to meet the eligibility criteria for approval.

The use of proportions, rather than absolute counts, ensures that the analysis is standardized across marital status groups. This approach accounts for the potentially different number of applicants within each group and allows for a clearer understanding of the likelihood of approval relative to marital status. By visualizing proportions, the plot highlights the approval trends without being skewed by differences in group sizes, making the insights more reliable and directly comparable.


### Industry
<img width="539" alt="Screenshot 2025-01-10 at 1 31 56 PM" src="https://github.com/user-attachments/assets/b783e563-8d0a-4135-b37e-7b250e9de749" />

This plot examines the trends in the proportion of credit approvals and rejections across various industries reported by applicants. While insightful, it is important to acknowledge that grouping applicants solely by industry may not fully capture their financial status. For example, a software engineer in the retail industry may have a financial profile more similar to other software engineers than to other retail workers. Grouping applicants by profession or role could provide a more nuanced understanding of their creditworthiness.

The plot shows that the largest proportion of applicants are from the energy sector, which also accounts for the highest number of approvals. However, the difference between the proportion of applicants rejected and the proportion approved indicates that the energy sector does not have the highest aggregate approval rate. On the other hand, applicants in the utilities industry exhibit a higher proportion of approvals relative to the total number of applicants from this sector. This suggests that workers in utilities may meet the credit approval criteria more consistently than applicants from other industries.

The use of proportions, rather than absolute counts, helps standardize the analysis by accounting for industry size variations. This method allows for a fairer comparison of approval rates across sectors, ensuring that the findings are not influenced by the number of applicants in each industry. By focusing on proportions, the plot provides clearer insights into how each industry performs relative to its size, making it easier to identify trends in credit approvals across different sectors.


### Gender
<img width="539" alt="Screenshot 2025-01-10 at 1 33 14 PM" src="https://github.com/user-attachments/assets/d12244bb-33e3-4030-9294-c96af67cceb5" />

This plot aims to investigate relationships between approval proportions and gender. The heat map suggests that generally fewer women apply as they have a lower aggregate proportion out of all of the applicants. While a larger proportion of men are approved compared to women, the larger proportion of men's applications are also rejected. This is also true for women although by a smaller margin. By a smaller margin, the proportion of women whose applications are rejected is higher than the proportion whose applications are accepted. This suggests that women generally have a higher approval rate compared to men, although this could be a result of factors like only high income women applying. This is a case where more information on the bank would have enriched my exploration. I would not expect such stark gender disparity from a metropolitan like Tokyo, although again the fact that this is a 1992 dataset could part of the reason behind this trend.
The use of proportions, rather than absolute counts, is important in this context as it helps to normalize for any imbalance in the number of male and female applicants. This ensures that the comparison between the approval rates of men and women is not skewed by the fact that men might make up a larger portion of the applicant pool. By using proportions, we can more accurately evaluate approval rates relative to the number of applicants from each gender, giving us a clearer picture of any potential disparities.


### Income
<img width="539" alt="Screenshot 2025-01-10 at 1 34 00 PM" src="https://github.com/user-attachments/assets/dbbbe502-1cfe-4011-adad-85d4dcf36484" />

This graph aims to illustrate the relationship between approval and income. To accommodate for the large range of income values I used a log transformation of the Income for my x-axis. The density plot for the approved group has a tall, narrow peak centered around the higher income levels, it is left-skewed, indicating that those with higher incomes are much more likely to be approved for credit. Interestingly the distribution for the rejected applicants much resembles a normal distribution. It seems as though just after 6 log(Income) the proportion of those approved is higher than the proportion rejected. However, at about 5 log(Income) the proportion of rejections peak and then begin to fall after that. There are outliers at both the very low and very high income ends. This suggests there may be unique factors influencing credit approval decisions for individuals at the extreme income levels. Since the income data is scaled, we cannot make any direct inferences from this except this generalization. Proportions are particularly useful here as they allow for a clearer comparison between approval and rejection rates, regardless of the absolute number of applicants in each income group. By focusing on proportions, we avoid being misled by the larger volume of applicants in certain income ranges, making it easier to assess trends across the entire dataset. This approach also normalizes the effect of income distribution, providing a more balanced view of the credit approval process across different income levels.


### Ethnicity
<img width="537" alt="Screenshot 2025-01-10 at 1 34 53 PM" src="https://github.com/user-attachments/assets/e5c816b5-9482-41df-9e9a-fad9cb1c55bd" />

This plot aims to show the relationship between approval proportions and reported ethnicity. The heatmap shows that the larger proportion of applicants report to be white. As a result, people of the White ethnicity have the highest proportion of approvals, but also a significantly higher proportion of rejections. I naturally found this surprising since Japan is an Asian country that has an overwhelming Asian majority. This could be because the Japanese themselved had not embraced the culture of credit in 1992, so most of the applicants were foreigners, but I could not find any information to support this hypothesis definitively. The Black ethnicity is the only one that shows a trend where there is a larger proportion of applicants approved compared to those whose applications are rejected. This is especially striking in comparison with the Latino ethnicity that has about the same proportion of rejected applicants but an even lower approval proportion. Using proportions here is important because it helps to normalize for the different numbers of applicants in each ethnic group. Without proportions, the conclusions might be misleading, especially since some groups (like White applicants) are over represented in the data. Proportions allow us to better understand the likelihood of approval relative to the number of applicants from each ethnic group, providing a clearer comparison across groups despite differences in sample size.


# Model Development

- The first step in this section was to split my data into a train and a test dataset. I used 70% of my data for training and then tested my model on the remaining 30% of the data.
- To choose my predictors, I first checked all the preditors for collinearity, and then conducted chi-squared tests for the various potential predictors and the status of either being approved for credit or rejected. Model Training:

From my exploration, I came up with 8 potential models influenced mostly by the multicolinearity of the indicators:

  - Gender, Married, DriversLicense, Income, YearsEmployed, zip_high, zip_med, Black, Latino, cit_other, cit_birth

  - Gender, Married, DriversLicense, Income, PriorDefault, zip_high, zip_med, Black, Latino, cit_other, cit_birth

  - Gender, Married, DriversLicense, Income, Employed, zip_high, zip_med, Black, Latino, cit_other, cit_birth

  - Gender, Married, DriversLicense, Income, CreditScore, zip_high, zip_med, Black, Latino, cit_other, cit_birth

  - Gender, BankCustomer, DriversLicense, Income, YearsEmployed, zip_high, zip_med, Black, Latino, cit_other, cit_birth

  - Gender, BankCustomer, DriversLicense, Income, PriorDefault, zip_high, zip_med, Black, Latino, cit_other, cit_birth

  - Gender, BankCustomer, DriversLicense, Income, Employed, zip_high, zip_med, Black, Latino, cit_other, cit_birth

  - Gender, BankCustomer, DriversLicense, Income, CreditScore, zip_high, zip_med, Black, Latino, cit_other, cit_birth

- Model 2, and 6 achieved ~0.87 classification rates. Both share the critical predictor PriorDefault, which captures an applicant's credit history. This feature is particularly impactful as it directly reflects past financial behavior, a strong indicator of creditworthiness. The inclusion of PriorDefault in both models reinforces its importance as a primary driver of classification accuracy, possibly making it the most important inicator for credit card approval within the context of this data. 
Interestingly, these models differ in their use of Married (Model 2) versus BankCustomer (Model 6). This suggests that the model could be indifferent about whether you use Married or BankCustomer which is very interesting because intuitively these provide very different pieces of information and this is likely true only within the context of this dataset.


### Results

My best model has a classification rate of about ~0.87. The results are in the confusion matrix below:

<img width="136" alt="Screenshot 2025-01-10 at 1 40 55 PM" src="https://github.com/user-attachments/assets/26ef440e-fa72-4dff-98ac-91d16019723e" />


### Model Assessment 
The baseline classifcation rate is 0.578. My models have classification rates of up to 0.87 which means they are capturing and leveraging some useful information to make the model robust. 


# Reflection
It was interesting to see how strongly multicollinerity affected my model right through to the end such that I could not objectively given my parameters for a best model within the constraints of this project decide which was best. I think this actually speaks to the redundancy of data financial institutions ask from the users of their services. It prompted me to reflect on whether more data means more information or we are grossly wasting resources to store and process the same information measured in different metrics. I think indeed, this would be an interesting application of this project, increasing data privacy, that is just asking less data from people but still retaining the most important information.

Furthermore, I felt that some categorical variables could be useful for my prediction and I ended up encoding them. The result was a modest increase in classification rates from about 0.82 to 0.87.  

My initial challenges with the data were just understanding its context. I think this was a perfect example to me for why context matters when dealing with data. When I downloaded it I had assumed it was from a US financial institution but as I explored it more I started realizing this was unlikely. I had to find information on it, and even then it was not really enough to explain the trends. While finding the year this dataset was uploaded did help, information on its location would have been key to my analysis as well.


# Citations
- Mann, R. J. (2001). Credit Cards and Debit Cards in the United States and Japan. SSRN Electronic Journal. https://doi.org/10.2139/ssrn.263009




