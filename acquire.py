"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.

Functions:
- get_repo_links
- get_header
- github_api_request
- get_repo_language
- get_repo_contents
- get_readme_download_url
- process_repo
- scrape_github_data
- get_readmes
"""

##### IMPORTS #####

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Union, cast
import requests
import random
import time
from bs4 import BeautifulSoup

# local
from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below or read from text file or get new from get_repo_links


# as a list
# REPOS = []


# read from text file
# opening the file in read mode
reps_file = open("repo_list.txt", "r")
# reading the file
reps = reps_file.read()
# replacing end splitting the text 
# when newline ('\n') is seen.
REPOS = reps.split("\n")


# use function to get fresh list
def get_repo_links():
    """
    This function scrapes GitHub search results for repositories with more than 0 stars and returns a
    list of their links.
    :return: a list of repository links from GitHub sorted by the number of stars in descending order.
    """
    # first url
    url = 'https://github.com/search?q=stars%3A%3E0&s=stars&type=Repositories&o=desc'
    # auth and username for headers
    # headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}
    # get first page contents
    search = requests.get(url,headers=get_header())
    # soup them up
    soup = BeautifulSoup(search.content,'html.parser')
    # get first page repos
    repo_list = [element.text for element in soup.find_all('a', class_='v-align-middle')]
    # get next page url
    next_url = 'https://github.com/'+[element['href'] for element in soup.find_all('a', class_='next_page')][0]
    # first sleep
    # time.sleep(5)
    # loop through each search page and get repos
    while next_url is not None:
                # get current page contents
                search_page = requests.get(next_url,headers=get_header())
                # get next page url
                next_url = 'https://github.com/'+[element['href'] for element in soup.find_all('a', class_='next_page')][0]
                # check if good
                if search_page.status_code == 200:
                    # make soup from current page
                    soup = BeautifulSoup(search_page.content, 'html.parser')
                    # extend/append repos to list
                    repo_list.extend(
                        iter([
                            element.text for element in soup.find_all(
                                'a', class_='v-align-middle')
                        ]))
                    # sleep some more
                    time.sleep(5)
                    continue
                else:
                    # if status no good, tell me
                    print(f'Error Code: {search_page.status_code}')
                    break
    return repo_list


def get_header():
    """
    The function returns a randomly selected user agent header for web scraping purposes.
    :return: a dictionary with a single key-value pair, where the key is 'User-Agent' and the value is a
    randomly chosen user agent string from a list of user agents.
    """
    # random list
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91",
        "Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/78",
        "Mozilla/5.0 (X11; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95",
        "Chrome/91.0.4472.12",
        "Mozilla/4.5 (compatible; HTTrack 3.0x; Windows 95)",
        "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/90",
        "Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/77",
        "Mozilla/5.0 (X11; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/94",
        "Chrome/91.0.4472.13",
        "Chrome/91.0.4472.132",
        "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/92",
        "Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/76",
        "Mozilla/5.0 (X11; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/93",
        "Chrome/91.0.4472.11",
        "Chrome/91.0.4472.10",
        "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/95",
        "Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/71",
        "Mozilla/5.0 (X11; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/96",
        "Chrome/91.0.4472.15",
        "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/98",
        "Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/80",
        "Mozilla/5.0 (X11; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/98",
        "Chrome/91.0.4472.16",
        "Chrome/91.0.4472.17",
        "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/99",
        "Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/79",
        "Mozilla/5.0 (X11; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/99",
        "Chrome/91.0.4472.18",
        "Chrome/91.0.4472.19",
        # github_username
        ]
    # pick random
    random_user_agent = random.choice(user_agents)
    return {'User-Agent': random_user_agent}


# headers for github api
headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

# checking before use
if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    """
    This function sends a request to the GitHub API and returns the response data as a list or
    dictionary, raising an exception if the response status code is not 200.
    
    :param url: The `url` parameter is a string that represents the URL of the GitHub API endpoint that
    we want to make a request to
    :type url: str
    :return: a JSON response from the GitHub API as either a list or a dictionary, depending on the
    structure of the response. If the response status code is not 200, an exception is raised with an
    error message that includes the status code and response data.
    """
    # basic api request
    response = requests.get(url, headers=headers)
    # make json
    response_data = response.json()
    # if no good return code
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    # return json
    return response_data


def get_repo_language(repo: str) -> str:
    """
    This function takes a GitHub repository name as input and returns the programming language used in
    the repository.
    
    :param repo: The `repo` parameter is a string that represents the name of a GitHub repository in the
    format "owner/repo_name"
    :type repo: str
    :return: The function `get_repo_language` returns the language of a given GitHub repository as a
    string. If the language is not specified for the repository, it returns `None`. If the API request
    does not return a dictionary, it raises an exception with an error message.
    """
    # repo url
    url = f"https://api.github.com/repos/{repo}"
    # get repo as json
    repo_info = github_api_request(url)
    # check if dict
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    """
    This function retrieves the contents of a GitHub repository and returns them as a list of
    dictionaries.
    
    :param repo: The `repo` parameter is a string that represents the name of a GitHub repository in the
    format "username/reponame"
    :type repo: str
    :return: The function `get_repo_contents` returns a list of dictionaries, where each dictionary
    contains information about a file or directory in the specified GitHub repository.
    """
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    # scrape data while also getting rid of any possible duplicates
    return [process_repo(repo) for repo in list(set(REPOS))]


def get_readmes():
    """
    This function checks if a JSON file exists, and if it does, reads it as a pandas dataframe,
    otherwise it scrapes data from GitHub, saves it as a JSON file, and returns the data as a dataframe.
    :return: a pandas DataFrame containing data from a local JSON file named "data2.json". If the file
    does not exist, the function first scrapes data from GitHub, saves it to the JSON file, and then
    returns the DataFrame.
    """
    # define filename
    filename = "data2.json"
    # check if it exists
    if not os.path.isfile(filename):
        # get data
        data = scrape_github_data()
        # dump to json locally
        json.dump(data, open(filename, "w"), indent=1)
        # return as df
        return pd.DataFrame(data)
    # get json
    else:
        # read json as df
        return pd.read_json(filename)


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)