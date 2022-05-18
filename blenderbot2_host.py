#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Example Flask server which hosts a model.

## Examples
**Serving the model**
python blenderbot2_host.py --model-file  /scratch/users/mali18/ParlAI/data/models/blenderbot2/blenderbot2_400M/model --search_server 0.0.0.0:8080 --n-docs 2

**Hitting the API***
```shell
curl -k http://172.20.35.75:80/response -H "Content-Type: application/json" -d '{"input": "Hello."}'
```
"""

from urllib import response
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
from flask import Flask, request
import os
import logging


class BBFlask(ParlaiScript):
    agent_list = {}
    save_dir = './agents_opt'

    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(True, True)
        return parser
    
    def init_agent_persona(self, agent, persona_path='agent_persona.txt'):
        with open(persona_path, 'r') as f:
            for line in f:
                print("line:", line)
                agent.observe({'text':f'Your persona: {line[-1]}', 'episode_done': False})
                response = agent.act()
                print(response['text'])
        return response

    def get_agent(self, user_id, reinit=True):
        if user_id not in self.agent_list.keys():
            # load if exists
            path = os.path.join(self.save_dir, user_id, 'model')
            if not reinit and os.path.exists(path):
                self.agent_list[user_id] = create_agent({"model_file":path})
            else:
                self.agent_list[user_id] = self.agent.clone()
                self.init_agent_persona(self.agent_list[user_id])
        return self.agent_list[user_id]

    def chatbot_response(self):
        data = request.json
        agent = self.get_agent(str(data['user_id']))
        agent.observe({'text': data["input"], 'episode_done': False})
        response = agent.act()
        return {'response': response['text']}

    def agent_persona(self):
        data = request.json
        agent = self.get_agent(str(data['user_id']))
        agent.observe({'text': f"Your persona: {data['input']}", 'episode_done': False})
        response = agent.act()
        return {'response': response['text']}

    def close_agent(self):
        try:
            data = request.json
            user_id = str(data['user_id'])
            logging.info(f"saving agent {user_id}")
            path = os.path.join(self.save_dir, user_id)
            os.makedirs(path, exist_ok=True)
            self.get_agent(user_id).save(f'{path}/model')
            del self.agent_list[user_id]
            return {'response': 'Success!'}

        except Exception as e:
            return {'reponse': 'Error 500', 'error_message': str(e)}

    def load_agent(self):
        try:
            data = request.json
            user_id = str(data['user_id'])
            reinit = data.get('reinit', True)
            logging.info(f"loading agent {user_id}. Reinit: {reinit}")
            self.get_agent(user_id, reinit=reinit)
            return {'response': 'Success'}

        except Exception as e:
            return {'reponse': 'Error 500', 'error_message': str(e)}

    def reset_agent(self,):
        try: 
            data = request.json
            user_id = data['user_id']
            logging.info(f"resent conversation for agent {user_id}")
            self.get_agent(user_id).reset()
            return {'response': 'Success!'}

        except Exception as e:
            return {'reponse': "Error 500", 'error_message': str(e)}
    
    def save_agents(self, ):
        for user_id in list(self.agent_list.keys()):
            logging.info(f"saving agent {user_id}")
            path = os.path.join(self.save_dir, user_id)
            os.makedirs(path, exist_ok=True)
            self.get_agent(user_id).save(f'{path}/model')

    def run(self):
        try:
            self.agent = create_agent(self.opt)
            app = Flask("parlai_flask")
            app.route("/response", methods=("GET", "POST"))(self.chatbot_response)
            app.route("/reset", methods=("GET", "POST"))(self.reset_agent)
            app.route("/closeConnection", methods=("GET", "POST"))(self.close_agent)
            app.route("/initiateConnection", methods=("GET", "POST"))(self.load_agent)
            app.route("/persona", methods=("GET", "POST"))(self.agent_persona)


            app.run(host='0.0.0.0', debug=True, port=8383, use_reloader=False)
        except:
            self.save_agents()


if __name__ == "__main__":
    BBFlask.main()
