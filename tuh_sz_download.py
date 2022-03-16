import multiprocessing as mp
import os
import re
import sys
from functools import partial

import requests
import tqdm
from requests.auth import HTTPBasicAuth

MAX_NUMBER_OF_CONNECTION = 1

# https://bramcohen.livejournal.com/70686.html
mylock = mp.Lock()
p = print


def print(*args, **kwargs):
    """Make print thread safe.

    Args:
        *args: multiple arguments.
        **kwargs: keyword arguments.
    """
    with mylock:
        p(*args, **kwargs)


def list_all_links_in_page(source: str):
    """Return all the urls in 'src' and 'href' tags in the source.

    Args:
        source: a strings containing the source code of a webpage.

    Returns:
        A list of all the 'src' and 'href' links
        in the source code of the webpage.
    """
    return re.findall(
        r'src\s*=\s*"([^"]+)"',
        source,
    ) + re.findall(
        r'href\s*=\s*"([^"]+)"',
        source,
    )


def is_link_displayed(link: str, source: str):
    """Check if the link is explicitly displayed in the source.

    Args:
        link: a string containing the link to find in the webpage source code.
        source: the source code of the webpage.

    Returns:
        True is the link is visible in the webpage, False otherwise.
    """
    return ('>' + link + '</a>') in source or (
        '>' + link[:link.find('.')]) in source


def scrap_page(
        url: str,
        username: str,
        password: str,
        files_queue,
        to_explore_queue):
    """Extract the url to explore and the url of the files to download.

    Args:
        url: a string which represents the URL of the webpage to download.
        username: the username to use for the authentication.
        password: the password to use for the authentication.
        files_queue: a Queue where the URLs of the files will be stored.
        to_explore_queue: a Queue whehe the URLs of the directories will
         be stored.
    """
    r = requests.get(url, auth=HTTPBasicAuth(username, password))

    if r.status_code == 401:
        print('Check the provided username and password.')
        return

    # Extract only the links displayed
    links = []
    for link in list_all_links_in_page(r.text):
        if is_link_displayed(link, r.text):
            links.append(link)

    # Put the files in the 'files_queue'
    # and the links in the 'to_explore_queue'
    for link in links:
        if '.' in link:  # It's a file
            files_queue.put(os.path.join(url, link).replace('\\', '/'))

        else:  # It's a folder
            to_explore_queue.put(os.path.join(url, link).replace('\\', '/'))


def list_files_links_and_paths(base_url: str, username: str, password: str):
    """Find all the files, paths and folders.

    1- Download the first page
    2- Find the useful link
    3- Download the useful link (while they are not files)
    4- Repeat until reaching all the files (when all link are explored)

    Args:
        base_url: a string which represents the URL of the base of the server.
        username: the username to use for the authentication.
        password: the password to use for the authentication.

    Returns:
        A tuple with sorted list of the links, the path_to_files,
        and folders
    """
    # Init the exploration list
    to_explore = [base_url]

    # Set the number of workers in the pool
    n_thread = 1.5 * os.cpu_count()

    print('Max number of workers used:', n_thread)
    manager = mp.Manager()
    q_files = manager.Queue()
    q_to_explore = manager.Queue()

    reduced_scrap = partial(
        scrap_page,
        username=username,
        password=password,
        files_queue=q_files,
        to_explore_queue=q_to_explore)

    n_exploration = 0  # Store the number of iteration
    while len(to_explore) > 0:
        # Display the cycle number and the number
        # of links to explore in the current cycle
        n_exploration += 1

        # Adjust the number of processes to use
        if sys.platform != 'win32':
            n_processes = min(len(to_explore), MAX_NUMBER_OF_CONNECTION)
        else:
            n_processes = min(
                min(len(to_explore), MAX_NUMBER_OF_CONNECTION),
                n_thread,
            )

        print(
            'Exploration cycle number ',
            n_exploration,
            ', ',
            len(to_explore),
            ' link(s) to explore (using ',
            n_processes,
            ' process(es)).',
            sep='')

        # Start pool
        chunksize = max(1, round(n_processes ** 0.5))
        with mp.Pool(processes=n_processes) as pool:
            for _ in tqdm.tqdm(
                pool.imap(
                    func=reduced_scrap,
                    iterable=to_explore,
                    chunksize=chunksize,
                ),
                total=len(to_explore),
            ):
                continue
        # Wait for the pool to join

        # Empty the to_explore list
        to_explore = []

        # Fill the to_explore list with the new values
        while not q_to_explore.empty():
            to_explore.append(q_to_explore.get())

    # Extract all the files in the queue and add them in the list
    files = []
    while not q_files.empty():
        files.append(q_files.get())

    if not len(files):
        # Wrong URL or authentication
        exit(1)

    print('Number of files:', len(files))

    if(base_url[-1] == '/'):
        base_url = base_url[:-1]

    # Return a sorted list with all the files link, one with the path,
    # relatively to the base URL and one with the folders
    # (without duplicate, "tree")
    return (
        sorted(
            list(
                set(
                    files
                ),
            ),
        ),
        sorted(
            list(
                {
                    filename[len(base_url):] for filename in files
                },
            ),
        ),
        sorted(
            list(
                {
                    filename[len(base_url): filename.rfind('/') + 1]
                    for filename in files
                },
            ),
        ),
    )


def make_all_dirs(dirs: list, base_path: str = None):
    """Make all the dirs for the dir in the dirs list.

    Can expand each dir with a base_path.

    Args:
        dirs: a list of directory's path to create.
        base_path: a base path to expand to the directory's path.
    """
    if base_path is None:
        base_path = os.getcwd()

    for a_dir in dirs:
        path = os.path.join(base_path, a_dir[1:])
        if not os.path.exists(path):
            os.makedirs(path)


def download_file(
        link_and_path_tuple: tuple,
        username: str,
        password: str,
        base_path: str = None):
    """Use the authentication and download the specified file
    to the specified path.

    Args:
        link_and_path_tuple: a tuple containing the URL
            of the file and the path to the file.
        username: the username to use for the authentication.
        password: the password to use for the authentication.
        base_path: a string containing the base path
            of where the files will be written.
    """
    if base_path is None:
        base_path = os.getcwd()

    link, path = link_and_path_tuple
    filename = os.path.basename(link)

    path = os.path.join(base_path, path[1:])

    try:
        if filename not in os.listdir(path[:path.rfind('/')]):
            print('Filename:', filename, '\nPath:', path, '\n')
            with requests.get(
                link,
                stream=True,
                auth=HTTPBasicAuth(username, password),
            ) as r:
                with open(path + '.part', 'wb') as filehandler:
                    for chunk in r.iter_content(chunk_size=4096):
                        filehandler.write(chunk)
            os.rename(path + '.part', path)
        else:
            print(
                'Filename:',
                filename,
                '(already downloaded)\nPath:',
                path,
                '\n')

    except OSError:
        print('Error:', sys.exc_info(), '(' + filename + ')')


def download_all(
        links: list,
        paths: list,
        base_path: str,
        username: str,
        password: str):
    """Download all the files and safe them in the path, using a base_path.

    Args:
        links: the list of all the files to download.
        paths: the list of all the file's path.
        base_path: a string containing the base path
                   of where the files will be written.
        username: the username to use for the authentication.
        password: the password to use for the authentication.
    """
    n_thread = 1.5 * os.cpu_count()

    # Adjust the number of processes to use
    if sys.platform != 'win32':
        n_processes = min(len(links), MAX_NUMBER_OF_CONNECTION)
    else:
        n_processes = min(
            min(len(links), MAX_NUMBER_OF_CONNECTION),
            n_thread,
        )

    print('Max number of workers used:', n_processes)

    links_and_paths = list(zip(links, paths))
    reduced_download_file = partial(
        download_file,
        username=username,
        password=password,
        base_path=base_path,
    )

    chunksize = max(1, round(n_processes ** 0.5))
    with mp.Pool(processes=n_processes) as pool:
        for _ in tqdm.tqdm(
            pool.imap(
                func=reduced_download_file,
                iterable=links_and_paths,
                chunksize=chunksize,
            ),
            total=len(links_and_paths),
        ):
            continue


def main(url: str, username: str, password: str, path: str = None):
    """Run the main functions to and download all the files.

    Args:
        url: a string which represents the URL of the base of the server.
        username: the username to use for the authentication.
        password: the password to use for the authentication.
        path: a string containing the path to where the files will be written.
    """
    links, path_to_files, folders = list_files_links_and_paths(
        url,
        username,
        password,
    )

    make_all_dirs(
        folders,
        base_path=path,
    )

    download_all(
        links,
        path_to_files,
        path,
        username,
        password,
    )


def file_count(path: str):
    """Return the number of file in a path and subpaths.

    Args:
        path: a string containing the path in which to count the files.

    Returns:
        The number of files int the path.
    """
    return sum(len(files) for r, d, files in os.walk(path))


if __name__ == '__main__':
    """
    dataset_version = "v1.5.2"
    path = "/content/" # "/content/drive/My Drive/Seizure_detection_project/"
    + dataset_version + "/TUH/"

    base_url = 'https://www.isip.piconepress.com/projects/
    tuh_eeg/downloads/tuh_eeg_seizure/' + dataset_version + '/edf'
    your_username = "nedc_tuh_eeg"
    your_password = "nedc_tuh_eeg"
    
    cmd 
  $python tuh_sz_download.py "https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.2/" nedc_tuh_eeg nedc_tuh_eeg --path /home_nfs/stragierv/TUH_SZ_v1.5.2/
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='TUH dataset downloader',
        description='Download all the folders and files under a base URL of'
                    ' the Picone dataset (https://www.isip.piconepress.com/'
                    'projects/tuh_eeg/downloads/, is the base URL'
                    ' to download everything).',
    )

    parser.add_argument(
        'URL',
        type=str,
        help='the base URL from which you want to start'
             ' to download the dataset',
        default= "https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.2/"
    )

    parser.add_argument(
        '-u',
        '--username',
        type=str,
        help='the username you got by email after filling the request'
             ' form (https://www.isip.piconepress.com/projects/'
             'tuh_eeg/html/request_access.php)',
        default="nedc_tuh_eeg"
    )

    parser.add_argument(
        '-p',
        '--password',
        type=str,
        help='the password you got by email after filling the request'
             ' form (https://www.isip.piconepress.com/projects/'
             'tuh_eeg/html/request_access.php)',
        default= "nedc_tuh_eeg"
    )

    parser.add_argument(
        '--path',
        type=str,
        nargs='+',
        help='by default the path is the current working directory,'
             ' but you can set it by yourself',
        default= "~/Downloads/thu/"
    )

    args = parser.parse_args()
    # args.username = "nedc_tuh_eeg"
    # args.password = "nedc_tuh_eeg"
    # args.path = "~/Downloads/thu/"


    urls = args.URL
    usr = args.username
    pwd = args.password
    paths = args.path

    if paths is None:
        paths = [None]

    # Paths can be a unique destination.
    # Or one path per URL or one path
    # per URL and current working directory for the rest.
    if len(paths) < len(urls):
        if len(paths) == 1:
            for _ in range(len(urls) - 1):
                paths.append(paths[0])
        else:
            for _ in range(len(urls) - len(paths)):
                paths.append(None)

    for url, path in zip(urls, paths):
        main(url=url, username=usr, password=pwd, path=path)

# Count the number of files
# find -type f|wc -l
