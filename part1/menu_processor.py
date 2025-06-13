import json

class MenuProcessor:
    def __init__(self, json_file_path):
        with open(json_file_path, encoding='utf-8') as f:
            self.menu_data = json.load(f)
        self.page_names = [item['page_name'] for item in self.menu_data]
        self.context_texts = [f"{item['Category']} {item['Service']} {' '.join(item['hierarchy'])}" for item in self.menu_data]
        self.full_texts = [f"{item['Category']} {item['Service']} {item['page_name']} {' '.join(item['hierarchy'])}" for item in self.menu_data]
    def get_menu_item(self, idx):
        return self.menu_data[idx]
    def calculate_weighted_similarity(self, full, page, context):
        return 0.4 * page + 0.4 * full + 0.2 * context 