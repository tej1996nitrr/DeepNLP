#script for model training
#%%
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

# def train_nlu(data, conf, model_dir):
#     training_data = load_data(data)
#     # trainer = Trainer(RasaNLUModelConfig(config))
#     trainer = Trainer(config.load(conf))
#     trainer.train(training_data)
#     model_directory = trainer.persist(model_dir, fixed_model_name='weather_nlu')
    
#     interpreter = Interpreter.load('./models/nlu/default/weather_nlu')
# if __name__ == "__main__":
#     train_nlu('F:/VSCode/DeepNLP/Rasa/weatherbot/data/data.json', 'F:/VSCode/DeepNLP/Rasa/weatherbot/config.json', 'F:/VSCode/DeepNLP/Rasa/weatherbot/models/nlu')
def train_nlu(data, config_file, model_dir):
    training_data = load_data(data)
    trainer = Trainer(config.load(config_file))
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir, fixed_model_name = 'weather_nlu')
    
if __name__ == '__main__':
    train_nlu('./data/data.json','config.json','./models/nlu')
# %%
