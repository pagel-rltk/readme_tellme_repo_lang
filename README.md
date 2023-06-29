# readme_tellme_repo_lang

# Presentation link
`<a href='https://www.canva.com/design/DAFnKQBwo7Q/Yl10_d24d1s0FLvJDG0-FA/view?utm_content=DAFnKQBwo7Q&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink'>`Presentation link`</a>`


## Project Planning

### Project Description

This project involves creating a machine learning model that uses natural language processing techniques to analyze README files and predict the main programming language used in the corresponding repositories. This tool could significantly streamline codebase management and analysis, particularly in large-scale scenarios.

### Project Goal

Develop a predictive model to identify GitHub repository's primary programming language using README file content.

### Initial Questions/Thoughts

If the README.md file has some of the below information we could point out the different programming languages.

- Find Keyword: Certain keywords might be more prevalent in README files of projects that use specific programming languages. For example, a README file for a Python project might frequently mention "pip", "PyPI", or "Jupyter", while a JavaScript project might mention "npm" or "Node.js". Finding the frequency of these keywords could provide clues about the primary programming language.
- Exploring the Library and Framework References: Projects often mention the libraries and frameworks they use, which are usually language-specific. For instance, a project mentioning "React" or "Express" is likely to be JavaScript-based, while "Pandas" or "Scikit-learn" would suggest Python.
- Looking at Code Snippets: Some README files include small code snippets or examples. If these are present, they can be a strong indicator of the programming language.
- Using Natural Language Processing (NLP): More advanced techniques could involve using NLP to understand the context and semantics of the text in the README file. This could potentially uncover more subtle indicators of the programming language.

## DATA Dictionary

| Feature         | Description                                                                             |
| --------------- | --------------------------------------------------------------------------------------- |
| repo            | The owners/organization and the repository name                                         |
| language        | The most common programming language in the repository                                  |
| top3other       | The top 3 most common programming languages across repositories and the rest are other  |
| readme_contents | The content inside the README.md file                                                   |
| clean           | The content after the prepare phase.                                                    |
| lemmatized      | The clean content after being lemmatized.                                               |

## The Plan

### Acquisition Phase

- The project commences with the acquisition of README files from a variety of repositories. Alongside these files, we also gather metadata about the repositories, which includes the primary programming language used. Our dataset included 960 repositories. We used GitHub's `<a href='https://github.com/search?q=stars%3A%3E0&s=stars&type=Repositories'>`the most starred repositories `</a>`. Generated the list of repositories programmatically using web scraping techniques.

### Preparation Phase

- Following acquisition, we move into the preparation phase. Here, the README text data underwent preprocessing, which encompasses cleaning, normalization, lemmatized, tokenization and removing stopwords. Simultaneously, the target variable, which is the primary programming language. Since the pages linked above change over time, note: "Our data comes from the top 960 most starred repositories on GitHub as of 27 June 2023".

### Exploration Phase

- Next, we enter the exploration phase. This involves conducting an exploratory data analysis on the README texts. The goal is to identify patterns, trends, and potential relationships between the text and the programming language used in the repository. Here we explored the following ideas:

> What are the most common words in READMEs?
> Does the length of the README vary by programming language?
> Do different programming languages use a different number of unique words?
> Are there any words that uniquely identify a programming language?

### Modeling Phase

- The final phase is modeling. The data is divided into training, validate and testing sets. Transform our content into a format suitable for a machine learning model. Try fitting several different models and using several different representations of the text (e.g., a simple bag of words, then also the TF-IDF values for each). Build a function that took in the text of a README file and attempted to predict the programming language. We used the top 3 languages (JavaScript, Objective-C, Java) and then label everything else as "Other" so that you have fewer unique values to predict. Each model's performance is evaluated on the testing set using appropriate metrics. The model that performs the best is then optimized and validated on unseen data.

## Steps to Reproduce

In order to get started reproducing this project, you'll need to set up a proper environment.

1. Create a new repository on GitHub to house this project.
2. Clone it into your local machine by copying the SSH link from GitHub and running **'git clone <SSH link'>** in your terminal.
3. Create a **.gitignore** file in your local repository and include any files you don't want to be tracked by Git or shared on the internet. You can create and edit this file by running **'code .gitignore'** or 'open .gitignore' in your terminal.
4. Make a github personal access token.

> a. Go here and generate a personal access token [https://github.com/settings/tokens](https://github.com/settings/tokens)
> You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
> b. Save it in your env.py file under the variable `github_token`
> c. Add your github username to your env.py file under the variable `github_username`
> d. Add more repositories to the `REPOS` list below.

5. Create a Jupyter Lab environment to continue working in. You can do this by running **'jupyter lab'** in your terminal.
6. In Jupyter Lab, create a new Jupyter Notebook to begin the data pipeline.

***Remember to regularly commit and push your changes to GitHub to ensure your work is saved and accessible from the remote repository.***
