import os
import json

class Logger(object):
	def __init__(self, name, folder='logs'):
		self.name = name
		self.folder = folder
		self.logs = {}
		if not os.path.exists(folder):
			os.mkdir(folder)
		self.fname = os.path.join(self.folder, self.name+'.json')

	def save(self):
		with open(self.fname, 'w') as f:
			json.dump([self.name, self.folder, self.logs], f)

	@staticmethod
	def load(fname):
		name, folder, logs = json.load(open(fname, 'r'))
		logger = Logger(name, folder=folder)
		logger.logs = logs
		return logger


	def log(self, name, value):
		if name not in self.logs: self.logs[name] = []
		self.logs[name].append(value)
		self.save()

	def set_value(self, name, value):
		self.logs[name] = value
		self.save()	

	def set_values(self, new_logs):
		self.logs.update(new_logs)
		self.save()		

