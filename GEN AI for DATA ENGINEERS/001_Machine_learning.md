# Machine Learning(Models will not understand text data)
## What is Machine Learning?

*   **Definition**: A field where computer models learn to perform tasks by analyzing data rather than following explicit instructions.
*   **Core Concept**: Data acts as the "fuel" for the machine learning "engine." Models use mathematical functions to identify patterns within this data to make predictions.

## Understanding Features and Labels

*   **Features**: The input variables or columns used by a machine learning model to make predictions. These represent the attributes or characteristics of the data (e.g., in a spam detector, features might include the presence of specific keywords or the number of links in an email).
*   **Labels**: The target variable or the outcome that the model is trying to predict (e.g., whether an email is categorized as "spam" or "not spam").
*   **Training Process**: In supervised learning, the model is provided with a dataset containing both the **features** and the corresponding **labels**. The model learns the relationship between these inputs and outputs.
*   **Prediction Phase**: Once training is complete, the model receives only the **features** and uses its learned internal weights to predict the corresponding **label**.

<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/e88ba533-5b19-4c81-ac2b-c8452d8c9ad5" />


## Train and Test Split
- Training data: 70 - 80% of total data, it should be diverse
- Test data: 20% of data is used and it doesn't know about training data

# Types of Machine Learning
## Supervised Learning
  - Classification: Like yes/no, we'll predict yes or no
    - ### *Logistic regression*
      - How Logistic Regression Works 
      *   **Purpose**: Despite its name, Logistic Regression is a **classification model** used to predict categorical outcomes (e.g., "spam" vs. "not spam").
      *   **Probability-Based Output**: The model calculates a probability score between 0 and 1:
          *   **Classification Threshold**: Typically, if the score is $ge 0.5$, it is classified as category "1" (e.g., spam); if $< 0.5$, it is classified as category "0" (e.g., not spam).
      *   **Mechanism**:
          1.  **Weight Assignment**: During training, the model analyzes input features and assigns specific **weights** based on their relative importance to the outcome.
          2.  **Score Calculation**: It calculates a total score by multiplying features by their respective weights and summing them.
          3.  **Sigmoid Function**: This calculated score is passed through the **sigmoid function**, a mathematical function that maps any numerical value (from negative to positive infinity) to a bounded range between 0 and 1.
      *   **Outcome**: The resulting value from the sigmoid function provides the final probability used to assign the class label.
    - ### *Decision Tree*
      A **Decision Tree** is a classification model that reaches predictions by navigating a series of **yes or no questions**.

        #### How it Works:
        *   **Hierarchical Decisions**: The model processes input features through a flow of logical gates. For example, it might first ask if an email contains the word "free," then follow up with a check on the number of links, eventually leading to a final classification of **spam** or **not spam**.
        *   **Logic Path**: Each node in the tree represents a feature test, and the branches represent the outcomes of those tests, guiding the model toward a final predicted label.

    - ### *Random Forest*
        #### How Random Forest Works
      
      *   **Definition**: A powerful **classification model** that functions as a collection of multiple *Decision Tree* models.
      *   **Training Process**:
          *   Multiple decision trees are trained individually on different, random subsets of the training data.
      *   **Prediction Mechanism**:
          *   When a new data point (e.g., an email) is introduced, every individual decision tree in the "forest" generates its own output prediction (e.g., "spam" or "not spam").
          *   The model uses a **majority vote** approach to determine the final result.
      *   **Core Concept**: Instead of relying on a single model's judgment, the system acts like a committee of experts, where the collective majority decision ensures higher accuracy and reliability.

    - ### Accuracy Assessment with Confusion Matrix
    
    *   **Purpose**: Used for evaluating the performance of **classification models** by comparing predicted labels against actual labels from a test dataset.
    *   **Components of the Matrix**:
        *   **True Positive (TP)**: Model correctly predicts a positive case (e.g., email is correctly identified as spam).
        *   **True Negative (TN)**: Model correctly predicts a negative case (e.g., email is correctly identified as not spam).
        *   **False Positive (FP)**: Model incorrectly predicts a positive case (e.g., non-spam email is incorrectly marked as spam).
        *   **False Negative (FN)**: Model incorrectly predicts a negative case (e.g., spam email is missed and marked as not spam).
    *   **Calculating Accuracy**: The formula is defined as:
        *   $\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Observations}}$
    *   **Example**: If a model makes 20 total predictions and gets 12 True Positives and 5 True Negatives, the accuracy is $\frac{12 + 5}{20} = 85%$.







    
  - ##Regression:
    In machine learning, **regression models** are a type of supervised learning algorithm used to predict **continuous numerical values** . Output will not be yes/no or a category but we predict some number or value
    
    Unlike classification models, which categorize data into discrete groups (e.g., spam vs. not spam), regression models estimate a specific value or quantity. Common applications include predicting the **price of a house** or the **stock price** on a given date.
        ## Linear and Polynomial Regression 
    
    *   **Regression Models Overview**:
    
        *   Used to predict **continuous numerical values** rather than discrete categories.
        *   Common use cases include predicting house prices or stock values.
    
    *   **Linear Regression**:
    
        *   Represents the relationship between features and the target as a **straight line**.
        *   Defined by the formula: $y = mx + c$
            *   $y$: Predicted output.
            *   $x$: Input feature.
            *   $m$: Slope of the line.
            *   $c$: Intercept.
    
    *   **Polynomial Regression**:
    
        *   Used to capture more complex, non-linear relationships by creating a **curved line** fit.
        *   Defined by higher-order formulas, such as: $y = m_1x + m_2x^2 + m_3x^3 + c$
     

## Evaluating Regression Models

Regression accuracy is measured by calculating how far the model's predictions deviate from the actual values. Common metrics include:

*   **Mean Absolute Error (MAE)**: The average of the absolute differences between predicted and actual values.
*   **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values.
*   **Root Mean Square Error (RMSE)**: The square root of the MSE.
*   **R-squared Score**: A statistical measure calculated as 1 minus the ratio of the MSE to the variance of the actual values.

## Unsupervised Learning

Unlike supervised learning, this approach uses data without pre-existing labels. The model identifies hidden patterns and structures independently.

*   **Clustering**: A primary technique for grouping data based on similarities.
*   **K-Means Clustering**: A popular algorithm that segments raw, unlabeled data into distinct clusters or groups based on feature patterns.

## Reinforcement Learning

This learning method involves an agent that improves through trial and error, rather than following a static labeled dataset.

*   **Feedback Mechanism**: The agent takes actions and receives feedback in the form of rewards for correct outcomes or penalties for errors.
*   **Goal Optimization**: The model iteratively adjusts its behavior to maximize its total reward, effectively learning an optimal strategy for navigation or decision-making.

## Curse of Dimensionality

This refers to the performance degradation that can occur when a dataset includes too many dimensions (features).

*   **Impact**: Adding excessive or irrelevant features increases the complexity of the data space, making it harder for the model to identify meaningful patterns and often reducing predictive accuracy.
*   **Dimensionality Reduction**: Techniques like Principal Component Analysis (PCA) are used to condense large numbers of features into a smaller, meaningful set, mitigating the curse while retaining critical information.

## Feature Engineering

This process involves preparing and transforming raw data into a format that machine learning models can process effectively.

*   **Rescaling (Normalization/Standardization)**: Transforms numerical features to a uniform range (such as 0 to 1) to prevent features with larger magnitudes from dominating those with smaller scales.
*   **One-Hot Encoding**: A method for converting categorical text data into binary numerical format by creating separate columns for each category.
*   **Handling Missing Data**: Strategies include removing incomplete records or filling in gaps using statistical measures like mean, median, or mode.
*   **Feature Extraction**: The practice of creating new, more meaningful features from existing data, such as deriving specific hours or days from a timestamp or extracting sentiment scores from text comments.
    
