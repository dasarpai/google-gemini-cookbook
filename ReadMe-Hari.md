Forked this repo from google-gemini-cookbook.
I am running these notebooks/expriments on google colab from my fork.
During experiments I will change certain setting, modify some code.
All the keys required for these experiments are stored in google-secret and they can be accessed using 

```py 
from google.colab import userdata
userdata.get('secretName')
```

Some of the experiment I will run on my local gpu. For the keys are kept in .env file. This file is included in .gitignore. Those who are using this work need to create .env file in the root folder. .env file has keys in following format.

OPENAI_API_KEY =sk-None-openaikey 
LANGSMITH_API_KEY=lsv2_langsmithkey
LLAMA_API_KEY=LL-Lamakey


To load these key in the environment for working you can use below python code.

```py 
!pip install python-dotenv

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access the variables
database_url = os.getenv('DATABASE_URL')
secret_key = os.getenv('SECRET_KEY')
```

