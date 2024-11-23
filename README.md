# Visual Image Processing Pipeline

## Overview

Leveraging [Ryven](https://ryven.org/) we have created a platform to facilitate building image
processing pipelines. We have added the "Visual Image Processing Pipeline" (VIPP)
functionality to Ryven. VIPP can be used for multidimensional image processing and was
developed in the context of fluorescence microscopy. We hope you find it useful. Enjoy!

## Dependencies

Install the required dependencies using `pip`:

```
pip install -r requirements.txt
```

## Instructions for Installing Ryven and VIPP (Windows)

Once you have cloned this repository (instructions below)

<!-- Note this needs to be cloned to site packages and we need to replace the run command to VIPP  -->

Ryven can be launched from the command line by typing `ryven`. If you installed Ryven into a Python virtual environment, the environment first needs to be activated.

When installed, VIPP will create a directory ~/.VIPP/ in your user home with the following structure:
(Note, VIPP is "Ryven" with adaptions)

```
VIPP
├──VIPP_example_projects
│   ├──sample_pipeline.json
│   └──sample_data.tiff
|
├──VIPP_projects
|   └──your_pipeline.json
|
├──VIPP_nodes
│   └──std
│       ├──nodes.py
│       ├──special_nodes.py
│       └──widgets.py
|
├──requiremnts.txt
|
└ ...
```

### 1. Install Python and Git

**Note:** The system has only been tested on Windows.

Ensure that both **Git** and **Python** are installed on your system:

- [Download Git](https://git-scm.com/downloads) and install it.
- [Download Python](https://www.python.org/downloads/) and install it. During installation, make sure to check the box to **Add Python to PATH**. This software uses Python 3 (VIPP has been tested with 3.10.0 as well as 3.10.15).

### 2. **Clone VIPP Git repository**

#### 2.1 **Open Command Prompt**

Open the Command Prompt by typing "cmd" in the Windows search bar and hitting Enter.

#### 2.2 **Navigate to the Desired Directory**

Use the `cd` command to navigate to the directory where you want to clone the **VIPP** repository. For example:

```
cd C:\path\to\your\projects
```

#### 2.3 **Clone the VIPP Repository**

Run the following command to clone the **VIPP** repository:

```
git clone https://github.com/EmmaSharratt/VIPP.git
```

This will create a new folder called `VIPP` in your current directory, containing the project files.

### 3. **Create the Virtual Environment (Optional)**

It is strongly encouraged to create a virtual environment for VIPP since it relies on specific versions of packages which might cause compatibility issues with other currently installed packages.

#### 3.1 **Open Command Prompt**

Open the Command Prompt. You can do this by typing "cmd" into the Windows search bar and hitting Enter.

#### 3.2 **Navigate to Your Project Directory**

Use the `cd` command to navigate to the folder where you want to create your virtual environment. For example:

```
cd C:\path\to\your\projects
```

#### 3.3 **Create the Virtual Environment**

To create a virtual environment, run the following command:

```
python -m venv my_venv_name
```

- Note you might need to substitute `python` for `python3.10` depending on how many versions of python you have installed.
- Replace `my_venv_name` with the name you want for your virtual environment. This will create a new folder with that name, containing the virtual environment files.

**Alternative**: If you have either **Anaconda** or **Miniconda** installed on your system (You can download Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html)).
Run the following command in Command Prompt or your preferred terminal (e.g., Anaconda Prompt):

```cmd
conda create --name my_venv_name python=3.10
```

- Conda will automatically download and install Python 3.10 from its repository.

#### 3.4 **Activate the Virtual Environment**

After creating the virtual environment, activate it with the following command:

```
.\venv_name\Scripts\activate
```

Once activated, you’ll see `(my_venv_name)` in your command prompt, indicating that the virtual environment is active.

**Alternative**: Using conda activate the environment with:

```cmd
conda activate my_venv_name
```

### 4. **Install Required Packages**

If you are using a virtual environment, ensure that it is activated.  
Command Prompt should look like this

```
(my_venv_name) C:\path\to\your\project\VIPP>
```

Now install packages:

```
pip install -r requirements.txt
```

This will install all dependencies listed in the `requirements.txt` file from the **VIPP** repository.

### 5. Run Ryven

Navigate to VIPP_example_projects into the VIPP folder you just cloned.  
Command Prompt should look like this:

```
(my_venv_name) C:\...\VIPP>
```

Enter "cd" (change directory) and your directory in the command prompt.

```
cd VIPP_example_projects
```

Run Ryven:

```
ryven
```

### 6. Create New Project

1. This will open a blank pipeline script
2. In Ryven window select `File>Import Nodes`
3. Navigate to `VIPP\VIPP_nodes\std\nodes.py` in the file browser and click open.
4. It may take about a minute for the nodes to load.
5. Start building your pipeline!

### 7. Load Example VIPP Project

1. Follow [5. Run Ryven](#5-run-ryven) steps
2. To load a VIPP sample project select “LOAD PROJECT”
3. Now select the sample project from the file browser (examples in VIPP_example_projects).
4. If it doesn't open immediately, and instead opens another dialog box, click on "ADD", and add the `VIPP\VIPP_nodes\std\nodes.py` file. Click "Ok". This may take a minute to load.
<!-- 4. Once the pipeline is open, load the sample image _<think about this>_
5. Check “confirm channel selection” -->

### 8. Save Your Pipeline

1. `File>Save Project` or shortcut `Ctrl + S`
2. Save file in `VIPP_projects` folder or desired location

<!-- ### 9. Load VIPP Project

Navigate to VIPP_projects into the VIPP folder you just cloned
Enter "cd" (change directory) and your directory in the command prompt.

```
cd path\to\...\VIPP\VIPP_projects
```

Follow `5. Run Ryven` steps (ie enter `ryven`)
Open your own `.json` file from the file browser.

Save projects here (`VIPP_projects`) -->

### 9. Deactivating the Virtual Environment

When you're done working, you can deactivate the virtual environment by running: `deactivate`. You will can reactivate at any stage using step 3.4. This keeps all your

## Contributors

Pipeline Software

- [Rensu Theart](https://rensu.co.za/)
- Emma Sharratt

Visual Image Procceing Pipeline Project

- Ben Loos
- Nicola Vahrmeijer

## To do

- [ ] Add article citations
