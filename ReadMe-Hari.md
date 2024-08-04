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

When the experiment is complete these notebooks cannot be saved in github with the same name (because of restriction on free account). <font color=red>
Therefore the file name will by updated if the notebook runs successfully, partially or not at all. -Hari-Runs-Fully, -Hari-Runs-Part, -Hari-Runs-Fail.</font>


# Expriment Status

1. google-gemini-cookbook\examples\Agents_Function_Calling_Barista_Bot.ipynb
1. google-gemini-cookbook\examples\Anomaly_detection_with_embeddings.ipynb
1. google-gemini-cookbook\examples\Apollo_11.ipynb
1. google-gemini-cookbook\examples\Classify_text_with_embeddings.ipynb
1. google-gemini-cookbook\examples\Guess_the_shape.ipynb
1. google-gemini-cookbook\examples\Market_a_Jet_Backpack.ipynb
1. google-gemini-cookbook\examples\Object_detection.ipynb
1. google-gemini-cookbook\examples\Opossum_search.ipynb
1. google-gemini-cookbook\examples\Search_reranking_using_embeddings.ipynb
1. google-gemini-cookbook\examples\Search_Wikipedia_using_ReAct.ipynb
1. google-gemini-cookbook\examples\Story_Writing_with_Prompt_Chaining.ipynb
1. google-gemini-cookbook\examples\Tag_and_caption_images.ipynb
1. google-gemini-cookbook\examples\Talk_to_documents_with_embeddings.ipynb
1. google-gemini-cookbook\examples\Upload_files_to_Colab.ipynb
1. google-gemini-cookbook\examples\Voice_memos.ipynb
1. google-gemini-cookbook\examples\chromadb\Vectordb_with_chroma.ipynb
1. google-gemini-cookbook\examples\json_capabilities\Entity_Extraction_JSON.ipynb
1. google-gemini-cookbook\examples\json_capabilities\Sentiment_Analysis.ipynb
1. google-gemini-cookbook\examples\json_capabilities\Text_Classification.ipynb
1. google-gemini-cookbook\examples\json_capabilities\Text_Summarization.ipynb
1. google-gemini-cookbook\examples\langchain\Chat_with_SQL_using_langchain.ipynb
1. google-gemini-cookbook\examples\langchain\Gemini_LangChain_QA_Chroma_WebLoad.ipynb
1. google-gemini-cookbook\examples\langchain\Gemini_LangChain_QA_Pinecone_WebLoad.ipynb
1. google-gemini-cookbook\examples\langchain\Gemini_LangChain_Summarization_WebLoad.ipynb
1. google-gemini-cookbook\examples\llamaindex\Gemini_LlamaIndex_QA_Chroma_WebPageReader.ipynb
1. google-gemini-cookbook\examples\prompting\Adding_context_information.ipynb
1. google-gemini-cookbook\examples\prompting\Basic_Classification.ipynb
1. google-gemini-cookbook\examples\prompting\Basic_Code_Generation.ipynb
1. google-gemini-cookbook\examples\prompting\Basic_Evaluation.ipynb
1. google-gemini-cookbook\examples\prompting\Basic_Information_Extraction.ipynb
1. google-gemini-cookbook\examples\prompting\Basic_Reasoning.ipynb
1. google-gemini-cookbook\examples\prompting\Chain_of_thought_prompting.ipynb
1. google-gemini-cookbook\examples\prompting\Few_shot_prompting.ipynb
1. google-gemini-cookbook\examples\prompting\Providing_base_cases.ipynb
1. google-gemini-cookbook\examples\prompting\Role_prompting.ipynb
1. google-gemini-cookbook\examples\prompting\Self_ask_prompting.ipynb
1. google-gemini-cookbook\examples\prompting\Zero_shot_prompting.ipynb
1. google-gemini-cookbook\examples\qdrant\Qdrant_similarity_search.ipynb
1. google-gemini-cookbook\quickstarts\Audio.ipynb
1. google-gemini-cookbook\quickstarts\Authentication.ipynb
1. google-gemini-cookbook\quickstarts\Authentication_with_OAuth.ipynb
1. google-gemini-cookbook\quickstarts\Caching.ipynb
1. google-gemini-cookbook\quickstarts\Code_Execution.ipynb
1. google-gemini-cookbook\quickstarts\Counting_Tokens.ipynb
1. google-gemini-cookbook\quickstarts\Embeddings.ipynb
1. google-gemini-cookbook\quickstarts\Error_handling.ipynb
1. google-gemini-cookbook\quickstarts\File_API.ipynb
1. google-gemini-cookbook\quickstarts\Function_calling.ipynb
1. google-gemini-cookbook\quickstarts\Function_calling_config.ipynb
1. google-gemini-cookbook\quickstarts\Gemini_Flash_Introduction.ipynb
1. google-gemini-cookbook\quickstarts\JSON_mode.ipynb
1. google-gemini-cookbook\quickstarts\Models.ipynb
1. google-gemini-cookbook\quickstarts\PDF_Files.ipynb
1. google-gemini-cookbook\quickstarts\Prompting.ipynb
1. google-gemini-cookbook\quickstarts\Safety.ipynb - **Hari-Runs-Fully**   
	model = genai.GenerativeModel('gemini-1.5-flash')   
	response = model.generate_content(unsafe_prompt)   
	response.candidates.finish_reason can be `FinishReason.STOP` or `FinishReason.SAFETY`   
	response.candidates[0].finish_reason   
	response.text   
	response = model.generate_content(
		unsafe_prompt,
		safety_settings={
			'HATE': 'BLOCK_NONE',
			'HARASSMENT': 'BLOCK_NONE',
			'SEXUAL' : 'BLOCK_NONE',
			'DANGEROUS' : 'BLOCK_NONE'
		})

response.candidates[0].finish_reason
1. google-gemini-cookbook\quickstarts\Streaming.ipynb
1. google-gemini-cookbook\quickstarts\System_instructions.ipynb
1. google-gemini-cookbook\quickstarts\Tuning.ipynb  **Hari-Runs-Fully**
	- Goto https://console.cloud.google.com/
	- Create google project
	- Create google billing account and attach payment details
	- Enable Generative Language API 
	- Create OAuth 2.0 Client IDs for Desktop client1 
	- Download secret_key.json 
	- Upload that file in the colab alongwith this notebook.
	- Install gcloud cli (Cloud SDK) on the desktop.
	- Go to command prompt and run following command 
		!gcloud auth application-default login --no-browser --client-id-file client_secret2.json --scopes='https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning'
	- It will generate one url. Copy that from the command prompt and paste in jupyter cell. With this authentication is complete.
	- Now you can run all the commands in of the notebook. If your credits are expired you may be charged for the running the commands.
	- ABOUT PROJECT:
		- Uses import google.generativeai as genai service 
		- Creates a finetuned model which can convert number written in any language into english digits.
		- finetuned model is stored in gemini and it cannot be download.
		- Using genai.create_tuned_model model we can create a fined tuned model. Instance can be assined to operation. It takes parameters. source_model=base_model.name, training_data=[
        {
             'text_input': '1',
             'output': '2',
        },{
             'text_input': '3',
             'output': '4',
        },...]
		id = name, epoch_count = 100, batch_size=4, learning_rate=0.001,)
		- finetuned model can be loaded from gemini into memory. genai.get_tuned_model(finetuned_modelname)
		- training process can be cancelled with operation.cancel()
		- history of training progress can be plot from the data. history = operation.result().tuning_task.snapshots
		- sns.lineplot(data=snapshots, x = 'epoch', y='mean_loss')
		- load model into memory, model = genai.GenerativeModel(model_name=f'tunedModels/{name}')
		- model prediction: model.generate_content('123455')
		- model description can be update: genai.update_tuned_model(f'tunedModels/{name}', {"description":"This is my model."});
		- model description can be read: model = genai.get_tuned_model(f'tunedModels/{name}') => model.description()
		- model can be deleted: genai.delete_tuned_model(f'tunedModels/{name}')
1. google-gemini-cookbook\quickstarts\Video.ipynb
1. google-gemini-cookbook\quickstarts\rest\Caching_REST.ipynb
1. google-gemini-cookbook\quickstarts\rest\Embeddings_REST.ipynb
1. google-gemini-cookbook\quickstarts\rest\Function_calling_config_REST.ipynb
1. google-gemini-cookbook\quickstarts\rest\Function_calling_REST.ipynb
1. google-gemini-cookbook\quickstarts\rest\JSON_mode_REST.ipynb
1. google-gemini-cookbook\quickstarts\rest\Models_REST.ipynb
1. google-gemini-cookbook\quickstarts\rest\Prompting_REST.ipynb
1. google-gemini-cookbook\quickstarts\rest\Safety_REST.ipynb
1. google-gemini-cookbook\quickstarts\rest\Streaming_REST.ipynb
1. google-gemini-cookbook\quickstarts\rest\System_instructions_REST.ipynb