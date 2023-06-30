# standard
import pandas as pd

# n-grams
import nltk
from wordcloud import WordCloud

# stats
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import kruskal

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

'''
*------------------*
|                  |
|     EXPLORE      |
|                  |
*------------------*
'''
# ---------------------------------Question 0 ----------------------------------------
# ------------------------------------------------------------------------------------
def get_normalized_value_counts(*args):
    """
    This function takes in multiple DataFrames and calculates the normalized value 
    counts for the 'top3other' column in each DataFrame. It returns a DataFrame 
    containing these value counts.

    Parameters:
    *args (DataFrame): Variable length argument, each being a DataFrame with a 
    'top3other' column.

    Returns:
    DataFrame: A DataFrame where each column represents a DataFrame and each row 
    represents a unique value from the 'top3other' column and its normalized count.
    """
    value_counts = {
        i: df.top3other.value_counts(normalize=True)
        for i, df in enumerate(args)
    }
    # Convert the dictionary to a DataFrame
    df_value_counts = pd.DataFrame(value_counts)

    # Rename columns
    df_value_counts.columns = ['train', 'val', 'test']

    return df_value_counts
# ---------------------------------Question 1 ----------------------------------------
# ------------------------------------------------------------------------------------



# ---------------------------------Question 2 ----------------------------------------
# ------------------------------------------------------------------------------------
def plot_readme_lengths(train):
    """
    This function takes in a DataFrame and plots a histogram of the 'length' column, 
    with different 'top3other' categories stacked. It also adds a vertical line at 
    the median 'length' and labels it.

    Parameters:
    train (DataFrame): The DataFrame containing the data to be plotted. It should 
    have columns 'length' and 'top3other'.

    Returns:
    None
    """
    # Create a stacked histogram of 'length', colored by 'top3other'
    sns.histplot(data=train, x='length', hue='top3other', multiple='stack')

    # Add a vertical line at the median 'length'
    plt.axvline(train.length.median(), color='pink', linestyle='--', linewidth=3)

    # Add a text label for the median line
    plt.text(x=3500, y=80, s='Median Length', color='hotpink')

    # Label the x and y axes
    plt.xlabel('Length of Readme')
    plt.ylabel('# of Readmes')

    # Add a title
    plt.title('Histogram of Readme Lengths hued on Language')

    # Display the plot
    plt.show()
# ------------------------------------------------------------------------------------    
def wilcox(train, cat_var, cat_value, quant_var, alt_hyp='two-sided'):
    x = train[train[cat_var]==cat_value][quant_var]
    y = train[quant_var].median()
    w = (x-y)
    # alt_hyp = ‘two-sided’, ‘less’, ‘greater’
    stat,p = stats.wilcoxon(w, alternative=alt_hyp)
    print("Wilcoxon Test:\n", f'stat = {stat}, p = {p}')
# ------------------------------------------------------------------------------------
def nova4(s1,s2,s3,s4):
    '''ANOVA test for 4 samples'''
    stat,p = stats.kruskal(s1,s2,s3,s4)
    print("Kruskal-Wallis H-Test:\n", f'stat = {stat}, p = {p}')
    from scipy.stats import wilcoxon

# ------------------------------------------------------------------------------------
def compare_readme_lengths(train, alpha=0.05):
    """
    This function takes in a DataFrame and a significance level, and performs a 
    Wilcoxon signed-rank test to compare the 'length' of each 'top3other' category 
    to the population median 'length'. It prints the results.

    Parameters:
    train (DataFrame): The DataFrame containing the data to be tested. It should 
    have columns 'top3other' and 'length'.
    alpha (float): The significance level for the Wilcoxon signed-rank test.

    Returns:
    None
    """
    # Get the population median 'length'
    median_length = train.length.median()

    # Perform the Wilcoxon signed-rank test for each 'top3other' category
    for lang in train.top3other.unique():
        print('|--------------------------------------')
        print(lang)
        stat, p = wilcoxon(train[train.top3other == lang].length - median_length)

        # Print the result
        if p < alpha:
            print(f'The median readme length of {lang} is significantly different than the population median readme length (p={p}).')
        else:
            print(f'The median readme length of {lang} is NOT significantly different than the population median readme length (p={p}).')

# ---------------------------------Question 3 ----------------------------------------
# ------------------------------------------------------------------------------------
def count_unique_words_by_language(train, p=True):
    """
    This function takes in a DataFrame and calculates the number of unique words used 
    in the README files of different programming languages. It prints these numbers.

    Parameters:
    train (DataFrame): The DataFrame containing the data to be analyzed. It should 
    have columns 'top3other' and 'lemmatized'.

    Returns:
    None
    """
    # Define a helper function to count unique words
    def count_unique_words(code):
        unique_words = set(code)
        return len(unique_words)

    # Get the words used in each programming language
    javascript = [word for row in train[train.top3other=='JavaScript']['lemmatized'] for word in row.split()]
    java = [word for row in train[train.top3other=='Java']['lemmatized'] for word in row.split()]
    objective_c = [word for row in train[train.top3other=='Objective-C']['lemmatized'] for word in row.split()]
    other = [word for row in train[train.top3other=='other']['lemmatized'] for word in row.split()]
    all_words = [word for row in train['lemmatized'] for word in row.split()]

    if p == True:
        # Print the number of unique words used in each programming language
        print("JavaScript unique words:", count_unique_words(javascript))
        print("Java unique words:", count_unique_words(java))
        print("Objective-C unique words:", count_unique_words(objective_c))
        print("Other unique words:", count_unique_words(other))
        print("All unique words:", count_unique_words(all_words))
    return javascript, java, objective_c, other, all_words
# ------------------------------------------------------------------------------------
def plot_unique_words_count(javascript, java, objective_c, other, all_words):
    """
    This function takes in a dictionary of unique words counts for different programming languages
    and plots a bar chart.

    Parameters:
    counts (dict): A dictionary where keys are programming languages and values are counts of unique words.

    Returns:
    None
    """
    # Define a helper function to count unique words
    def count_unique_words(code):
        unique_words = set(code)
        return len(unique_words)
    
    # the counts the words used in each programming language
    counts = {
    'JavaScript': count_unique_words(javascript),
    'Java': count_unique_words(java),
    'Objective-C': count_unique_words(objective_c),
    'Other': count_unique_words(other),
    'All Words': count_unique_words(all_words)
    }
    
    # Create bar chart
    plt.bar(counts.keys(), counts.values(), color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title('Unique Words Count for Each Programming Language')
    plt.xlabel('Programming Language')
    plt.ylabel('Unique Words Count')
    plt.xticks(rotation=45)
    plt.show()


# ---------------------------------Question 4 ----------------------------------------
# ------------------------------------------------------------------------------------
def analyze_unique_words(*args):
    """
    This function takes in multiple lists of words, generates n-grams for each list, 
    identifies the common and unique words among the lists, and returns dictionaries 
    of unique words and their counts for each list.

    Parameters:
    *args (list): Variable length argument, each being a list of words.

    Returns:
    dict: A dictionary where keys are the index of the input list and values are 
    dictionaries of unique words and their counts.
    """
    # Initialize dictionaries to store n-grams and unique words
    ngram_dicts = {}
    unique_word_dicts = {}

    # Generate n-grams for each list of words
    for i, words in enumerate(args):
        ngrams = pd.Series(nltk.ngrams(words, 1)).value_counts().head(20) # add a head of 20 if needed after value_counts
        ngram_dicts[i] = {k[0]+' ': v for k, v in ngrams.to_dict().items()}

    # Get sets of words for each language
    word_sets = {i: set(ngram_dict.keys()) for i, ngram_dict in ngram_dicts.items()}

    # Find common words
    common_words = set.intersection(*word_sets.values())

    # Find and store unique words for each list
    for i, word_set in word_sets.items():
        unique_words = word_set - common_words
        unique_word_dicts[i] = {key: ngram_dicts[i][key] for key in unique_words}

    return unique_word_dicts

# ------------------------------------------------------------------------------------
def conv_dict_to_df(unique_word_dicts):
    # Convert dictionaries to DataFrame
    df = pd.DataFrame(unique_word_dicts)
    df.columns = [f'List_{str(i)}' for i in df.columns]

    # Rename columns
    df.columns = ['uni_javascript', 'uni_java', 'uni_objective_c', 'uni_other']
    return df

# ------------------------------------------------------------------------------------
def create_word_clouds(uni_javascript, uni_java, uni_objective_c, uni_other):
    """
    This function creates and displays word clouds for four different sets of words.
    The word clouds are displayed in a 2x2 grid.

    Parameters:
    uni_javascript (dict): A dictionary where keys are words and values are their frequencies in JavaScript.
    uni_java (dict): A dictionary where keys are words and values are their frequencies in Java.
    uni_objective_c (dict): A dictionary where keys are words and values are their frequencies in Objective-C.
    uni_other (dict): A dictionary where keys are words and values are their frequencies in other languages.
    """

    # Create a 2x2 subplot with specific size
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Generate word clouds for each language
    # The WordCloud function generates a word cloud image from the word frequencies
    words_img_javascript = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(uni_javascript)
    words_img_java = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(uni_java)
    words_img_objective_c = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(uni_objective_c)
    words_img_other = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(uni_other)

    # Display the word cloud for JavaScript
    axs[0, 0].imshow(words_img_javascript)
    title = axs[0, 0].set_title('JavaScript', fontsize=20, color='white')
    axs[0, 0].axis('off')  # Hide the axis
    plt.setp(title, backgroundcolor='red')  # Set the title background color to red

    # Display the word cloud for Java
    axs[0, 1].imshow(words_img_java)
    title = axs[0, 1].set_title('Java', fontsize=20, color='white')
    axs[0, 1].axis('off')
    plt.setp(title, backgroundcolor='red')

    # Display the word cloud for Objective-C
    axs[1, 0].imshow(words_img_objective_c)
    title = axs[1, 0].set_title('Objective-C', fontsize=20, color='white')
    axs[1, 0].axis('off')
    plt.setp(title, backgroundcolor='red')

    # Display the word cloud for other languages
    axs[1, 1].imshow(words_img_other)
    title = axs[1, 1].set_title('Other', fontsize=20, color='white')
    axs[1, 1].axis('off')
    plt.setp(title, backgroundcolor='red')

    # Adjust the layout so that there is no overlap between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()