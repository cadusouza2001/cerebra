# scrape_fast_resumable.py (Versão Final com Filtro de API)

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time
import os

# --- Configurações e Nomes de Arquivos ---
BASE_URL = "https://spark.apache.org/docs/latest/"
START_URL = BASE_URL + "index.html"
OUTPUT_FILE = "spark_documentation_guides.jsonl" # Novo nome para refletir o conteúdo filtrado
VISITED_URLS_FILE = "visited_urls_guides.log"
MAX_CONCURRENT_REQUESTS = 100

# --- Armazenamento Compartilhado ---
visited_urls = set()
page_counter = 0

def load_visited_urls():
    """Carrega os URLs já visitados de um arquivo de log, se ele existir."""
    if not os.path.exists(VISITED_URLS_FILE):
        return set()
    
    print(f"Carregando URLs já visitados de '{VISITED_URLS_FILE}'...")
    with open(VISITED_URLS_FILE, 'r', encoding='utf-8') as f:
        visited = set(line.strip() for line in f)
    print(f"Encontrados {len(visited)} URLs já visitados. Eles serão ignorados.")
    return visited

async def worker(session, queue, lock):
    """Pega uma URL da fila, processa e adiciona novos links de volta à fila, ignorando /api/."""
    global page_counter
    while True:
        try:
            url = await queue.get()
            canonical_url = url.split('#')[0]

            # Adquire o lock ANTES de verificar e modificar a lista de visitados
            async with lock:
                # A verificação agora inclui o filtro para ignorar links de API
                if (not canonical_url or 
                    not canonical_url.startswith(BASE_URL) or 
                    "/api/" in canonical_url  or "/sql-ref-" in canonical_url or
                    canonical_url in visited_urls):
                    queue.task_done()
                    continue
                
                # Marca como visitado IMEDIATAMENTE para evitar que outros workers o peguem
                visited_urls.add(canonical_url)
            
            # Agora que o URL está "reivindicado", podemos fazer a operação de rede demorada
            print(f"Rastreando: {canonical_url}")
            try:
                async with session.get(canonical_url, timeout=15) as response:
                    if response.status == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        if soup.body:
                            text = soup.body.get_text(separator='\n', strip=True)
                            if len(text.strip()) > 200:
                                item_to_save = {"url": canonical_url, "content": text}
                                async with lock:
                                    # Salva o conteúdo e o progresso
                                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                                        f.write(json.dumps(item_to_save, ensure_ascii=False) + '\n')
                                    with open(VISITED_URLS_FILE, 'a', encoding='utf-8') as f:
                                        f.write(canonical_url + '\n')
                                    page_counter += 1
                                print(f"  └─ [SUCESSO] Conteúdo salvo. ({page_counter} páginas de guias salvas)")
                        
                        # Adiciona novos links (já filtrados) à fila
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            full_url = urljoin(canonical_url, href)
                            await queue.put(full_url)
            except Exception as e:
                print(f"  └─ [FALHA] Erro ao processar {canonical_url}: {e}")

            queue.task_done()
        except asyncio.CancelledError:
            break

async def main():
    """Função principal que orquestra o processo de scraping assíncrono."""
    global visited_urls
    visited_urls = load_visited_urls()
    
    if not visited_urls:
        print("Primeira execução detectada. Criando novos arquivos de log e saída...")
        open(OUTPUT_FILE, 'w').close()
        open(VISITED_URLS_FILE, 'w').close()

    queue = asyncio.Queue()
    lock = asyncio.Lock()
    await queue.put(START_URL)

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(worker(session, queue, lock)) for _ in range(MAX_CONCURRENT_REQUESTS)]
        await queue.join()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()

    print("\n" + "="*50)
    print("Rastreamento com Filtro de API Completo!")
    print(f"Tempo total: {end_time - start_time:.2f} segundos")
    print(f"Total de páginas de guias salvas nesta sessão: {page_counter}.")
    print(f"Os resultados estão em '{OUTPUT_FILE}' e o progresso em '{VISITED_URLS_FILE}'.")