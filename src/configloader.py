import json
from pathlib import Path

class NestedDicet:
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, NestedDicet(value)) # 递归调用
            else :
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

class ConfigLoader(NestedDicet):
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        with open(config_path,"r",encoding= "utf-8") as f:
            config_dict = json.load(f)
        super().__init__(config_dict)
        self.config_dict = config_dict