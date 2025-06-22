import json
import os
from typing import List, Dict, Any

class MenuDataLoader:
    """메뉴 데이터를 로드하고 전처리하는 클래스"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.menu_data = []
        self.menu_names = []
        
    def load_data(self) -> bool:
        """메뉴 데이터를 로드합니다."""
        try:
            if not os.path.exists(self.data_path):
                print(f"메뉴 데이터 파일을 찾을 수 없습니다: {self.data_path}")
                return False
                
            with open(self.data_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                
            # 데이터 구조에 따라 처리
            if isinstance(data, list):
                self.menu_data = data
            elif isinstance(data, dict) and 'menu' in data:
                self.menu_data = data['menu']
            else:
                print("지원하지 않는 데이터 형식입니다.")
                return False
                
            # 메뉴 이름 추출
            self.menu_names = self._extract_menu_names()
            print(f"총 {len(self.menu_names)}개의 메뉴를 로드했습니다.")
            return True
            
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {e}")
            return False
    
    def _extract_menu_names(self) -> List[str]:
        """메뉴 데이터에서 메뉴 이름들을 추출합니다."""
        menu_names = []
        
        for item in self.menu_data:
            if isinstance(item, dict):
                # 다양한 키 이름으로 메뉴 이름 찾기
                menu_name = None
                for key in ['name', 'menu_name', 'title', 'menu', 'item', 'page_name']:
                    if key in item and item[key]:
                        menu_name = str(item[key]).strip()
                        if menu_name and menu_name != " ":
                            break
                
                if menu_name:
                    menu_names.append(menu_name)
            elif isinstance(item, str):
                menu_names.append(item)
        
        return list(set(menu_names))  # 중복 제거
    
    def get_menu_list_text(self, max_items: int = 100) -> str:
        """메뉴 목록을 텍스트 형태로 반환합니다."""
        if not self.menu_names:
            return "메뉴 데이터가 없습니다."
        
        # 최대 개수만큼만 사용
        display_items = self.menu_names[:max_items]
        
        menu_text = "\n".join([f"- {name}" for name in display_items])
        
        if len(self.menu_names) > max_items:
            menu_text += f"\n... (총 {len(self.menu_names)}개 중 {max_items}개 표시)"
        
        return menu_text
    
    def get_all_menu_names(self) -> List[str]:
        """모든 메뉴 이름을 반환합니다."""
        return self.menu_names.copy()
    
    def get_menu_data(self) -> List[Dict[str, Any]]:
        """전체 메뉴 데이터를 반환합니다."""
        return self.menu_data.copy() 