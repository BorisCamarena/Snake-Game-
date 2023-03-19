
# Importamos las librerias

# Cerebro del agente -> Pytorch.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

 # Red Neuronal Lineal Q

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# Optimizacion.

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
# Entrenamiento - Un paso.


    def train_step(self, state, action, reward, next_state, done):

	    # Vectores -> Tensores.
	    
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        # Aprendizaje corto plazo.

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: Prediccion de valores Q -> Estado actual.
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:

		    # Ecuacion de Bellman -> Calidad de accion.
		    
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

                # El Q_new se guarda en la matriz target(len(done)x3)

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> si no a terminado.
        # pred.clone()
        # preds[argmax(action)] = Q_new

        # Esta parte es muy interesante en como se comporta el gradiente
        # por que no se para en los puntos extremos de R-F ( Region de
        # factibilidad ) si no que avanza mas.
        # Seria bueno el estudio utilizando
        # Metodo simplex.
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        # Calcula el gradiente.
        loss.backward()
        # Optimiza ( mìnimo ) en lugar de un màximo.
        # Serìa bueno analizar, si el modelo esta en la forma estandar.

        self.optimizer.step()


