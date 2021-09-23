import yaml

class DBConfig(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_tag = u'!DBConfig'
    def __init__(self, connector: str, server: str, port: int, username: str, password: str, database: str):
        self.connector = connector
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.database = database
    
    @property
    def connection(self):
        return f"{self.connector}://{self.username}:{self.password}@{self.server}:{self.port}/{self.database}"

    def __repr__(self):
        return f"{self.__class__.__name__}(connector={self.connector}, server={self.server}, port={self.port}, username={self.username}, password={self.password}, database={self.database}"


class ConfigManager():
    def __init__(self, config_file = 'config.yml'):
        with open(config_file, 'r') as stream:
            self.config: DBConfig = yaml.safe_load(stream)

config_manager = ConfigManager()