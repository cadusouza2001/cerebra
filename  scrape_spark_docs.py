# scrape_spark_docs.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time

# O ponto de partida para o nosso rastreamento
BASE_URL = "https://spark.apache.org/docs/latest/"
# Um conjunto para acompanhar os URLs visitados para evitar loops infinitos
visited_urls = set()
# Uma lista para armazenar o conteúdo da nossa documentação
documentation_content = []

def scrape_page(url):
    """Rastreia uma única página, extrai o conteúdo e encontra novos links."""
    if url in visited_urls:
        return

    print(f"Rastreando: {url}")
    visited_urls.add(url)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Lança uma exceção para códigos de status ruins
        soup = BeautifulSoup(response.content, 'html.parser')

        # Encontra a div de conteúdo principal. Isso pode precisar de ajuste se a estrutura do site mudar.
        # Inspecionar a página mostra que o conteúdo principal está dentro de uma <div role="main">
        main_content = soup.find('div', role='main')

        if main_content:
            # Extrai o texto e o limpa
            text = main_content.get_text(separator='\n', strip=True)
            documentation_content.append({"url": url, "content": text})

        # Encontra todos os links na página e os segue se fizerem parte da documentação
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Cria URL absoluto
            full_url = urljoin(BASE_URL, href)

            # Segue o link apenas se estiver dentro do mesmo site de documentação
            if full_url.startswith(BASE_URL) and full_url not in visited_urls:
                # Um pequeno atraso para ser educado com o servidor
                time.sleep(0.1)
                scrape_page(full_url)

    except requests.RequestException as e:
        print(f"Erro ao rastrear {url}: {e}")

# Inicia o processo
scrape_page(BASE_URL + "index.html")

# Salva o conteúdo rastreado em um arquivo
with open('spark_documentation_raw.json', 'w', encoding='utf-8') as f:
    json.dump(documentation_content, f, indent=4, ensure_ascii=False)

print(f"\nRastreamento completo. Encontradas {len(documentation_content)} páginas.")
print("Conteúdo bruto salvo em spark_documentation_raw.json")