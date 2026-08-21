# Estratégias de otimização orientadas pelo PGO

Este documento cruza o perfil LLVM da nova execução sem logging com os caminhos reais da codebase e com a telemetria diagnóstica da execução anterior. O objetivo é ordenar intervenções que possam reduzir o frametime de CPU do GOW3 sem trocar uma espera por uma regressão de sincronização, ordem de PM4 ou estado Vulkan incorreto.

## Escopo e qualidade dos dados

O novo raw é:

`Build/x64-Clang-Ultra/profiles/raw/shadps4-31412-10048391627949475423_0.profraw`

Ele foi produzido com a correção de flush do perfil e é lido normalmente pelo LLVM. A execução foi feita com `Enable Logging` desligado, portanto não há um CSV novo de telemetria para essa run. A telemetria usada como contexto é a run anterior com logging, em:

`C:\Users\Arthur\AppData\Roaming\shadPS4\log\shadps4-telemetry-1785640362686.csv`

Isso importa por dois motivos:

1. O raw LLVM contém contagens de execução, não ciclos de CPU, stalls ou tempo por função.
2. Os contadores absolutos não podem ser comparados como se as duas execuções tivessem a mesma duração. O novo perfil tem aproximadamente 1,9–2,3 vezes as contagens nos caminhos principais, o que é compatível com uma janela/carga diferente. A comparação válida aqui é a permanência dos mesmos caminhos no topo e a proporção entre eles.

O perfil sem logging ainda mostra `PerformanceTelemetry::Enabled`/`Log::IsEnabled` e o wrapper `PerformanceTelemetry::Add` como hotpaths. Isso é apenas o custo da checagem gateada; com o logging desligado o ring buffer não é escrito. Assim, os hotpaths abaixo não são artefatos do ring de telemetria.

## O que mudou entre os perfis

| Caminho | Raw anterior (logging) | Raw novo (sem logging) | Leitura correta |
|---|---:|---:|---|
| `Liverpool::ProcessCommands` | 72,3 M | 146,6 M | O dispatch de callbacks continua no caminho crítico; escala com o trabalho. |
| `Liverpool::ProcessGraphics` / `NextPacket` | 51,9 M | 99,0 M | Decode PM4 e acesso repetido a header/bitfields continuam dominantes. |
| `PM4CmdWaitRegMem::Test` | 20,4 M | 47,5 M | A espera ainda reavalia o mesmo predicado muitas vezes. |
| `PsColorBuffer::operator==` | 143,1 M | 285,8 M | Comparações profundas de runtime/pipeline são um hotpath real. |
| `PsInput::operator==` | 101,3 M | 202,4 M | Mesmo sinal para especialização de fragment shader. |
| `ImageSpecialization::operator==` | 41,5 M | 80,2 M | Busca de módulo/pipeline repete igualdade de estruturas grandes. |
| `CompMapping::operator==` | 38,0 M | 74,8 M | Comparação de recursos ocorre com alta frequência mesmo com poucas misses de pipeline. |
| `TextureCache::TouchImage` | 45,4 M | 87,4 M | LRU/mutex da texture cache aparece em quase todos os draws. |
| `Image::GetBarriers` | 25,6 M | 50,3 M | Planejamento de transições por subrecurso permanece frequente. |
| `UpdateViewportScissorState` | 30,5 M | 57,9 M | Estado dinâmico é recalculado por draw. |
| `CopyMappedBytes` | 128,6 M | 197,2 M | Upload/cópia é relevante; o caminho já tem AVX2 e precisa de ajuste baseado em tamanhos. |
| `ExecuteStreamCopyBatch` (lambda) | 32,9 M | 63,4 M | Coalescência/deduplicação de stream copies é quente, mas já possui fastpath unitário. |
| `MasterSemaphore::CurrentTick` | 19,0 M | 36,3 M | Polling de timeline é recorrente e deve ser amortizado, não removido sem prova. |

Na run anterior, 12.302 pacotes `WaitRegMem` geraram 20,36 milhões de chamadas a `Test`, cerca de 1.655 avaliações por pacote. O novo perfil confirma a estrutura: `Test` e `Liverpool::Process` sobem juntos. Essa é a evidência mais forte para atacar a forma de espera antes de micro-otimizar bitfields.

Na mesma run diagnóstica, houve 51,75 milhões de pacotes PM4, 1,90 milhão de draws e 54.067 submits gráficos. O GCP teve 63,60 s ativo e 2,76 s bloqueado, enquanto foram observados 4.820 gaps de idle da GPU totalizando 59,91 s. Esses valores são de uma execução com logging e não devem ser usados como benchmark absoluto, mas apontam para um produtor que passa muito tempo decodificando, esperando ou submetendo trabalho pequeno.

## Ranking de alvos

### 1. Esperas `WaitRegMem` e escalonamento do GCP — prioridade P0

#### Evidência e caminho

O caminho é `Liverpool::Process` → `ProcessGraphics`/`ProcessCompute` → `PM4CmdWaitRegMem::Test` em `src/video_core/amdgpu/pm4_cmds.h`. Hoje o loop lê novamente espaço, endereço/registro, máscara, referência e função a cada iteração; quando a condição não está pronta, `YIELD_GFX()` retorna ao escalonador. Em `liverpool.cpp`, o escalonador percorre filas e retoma coroutines uma por uma.

#### Estratégia

1. Decodificar o comando uma vez ao entrar no pacote: `mem_space`, `function`, `mask`, `reference` e, para memória, o endereço efetivo. Criar um predicado pequeno e imutável no frame da coroutine. Manter `PM4CmdWaitRegMem::Test` como fallback exato para formatos incomuns.
2. Especializar os casos frequentes (`Memory`, `Equal`/`NotEqual`, máscara cheia e máscara parcial) com funções sem extração repetida de `BitField`. Não fazer `always_inline` global: o ganho pretendido é eliminar trabalho repetido, não aumentar o código quente.
3. Substituir a reavaliação por um `WaitToken` indexado por endereço/página e tipo de fonte. A escrita de label/timeline marca tokens interessados como prontos e acorda o owner da fila. O token precisa conter a geração/epoch do recurso para descartar um wakeup obsoleto.
4. Separar filas prontas e bloqueadas. Um `ready_mask`/`blocked_mask` compacto e uma fila de eventos tornam o caminho pronto `countr_zero(mask)` em vez de varrer até 57 filas. A fila bloqueada só é visitada quando o endereço observado muda.
5. Acordar por lote: várias escritas do mesmo submit devem produzir um único wakeup e uma única retomada. A ordem dos submits e o `rptr` continuam sendo a autoridade; o token apenas evita polls inúteis.

#### Instrumentação e gates

Adicionar contadores gateados: `wait_token_hits`, `wait_token_misses`, `wait_predicate_evals`, `wait_wake_coalesced`, `wait_stale_epoch`, `ready_mask_picks` e `blocked_index_lookups`. Registrar amostras de opcode/endereço somente em um ring de diagnóstico de baixa frequência.

O gate é: zero perda de wakeup, zero avanço de `rptr` antes da condição, progresso em espera infinita válida, mesma sequência de PM4/IB, nenhuma regressão visual e redução comprovada de `wait_predicate_evals` por pacote. O ganho deve aparecer no frametime final; reduzir a contagem de polls sem reduzir o tempo de frame não basta.

### 2. Caminho quente de pipeline e especialização — prioridade P1

#### Evidência e caminho

`Liverpool::ProcessGraphics` chama `Rasterizer::Draw`; `vk_rasterizer.cpp` chama `PipelineCache::GetGraphicsPipeline`; `vk_pipeline_cache.cpp` executa `RefreshDynamicProgramData`, `ResolveStageResources`, `EqualResolvedResources` e buscas de módulos. Mesmo com apenas quatro misses de pipeline na telemetria anterior, `PsColorBuffer`, `PsInput`, `ImageSpecialization`, `CompMapping` e `ResolveResourceList` aparecem entre os maiores contadores. Isso caracteriza um warm path caro, não compilação de pipeline.

#### Estratégia

1. Dar prioridade ao slot atualmente usado. `StageOptimizationSet::current` hoje é invalidado e a lista de entradas é percorrida. Validar primeiro `(program_hash, base_hash, raw_dependency_generation, runtime_info_fingerprint, resource_generation)`; somente em falha executar a igualdade completa.
2. Calcular fingerprints baratos e estáveis para `FragmentRuntimeInfo`, `StageSpecialization` e arrays de dependências. O fingerprint é apenas pré-filtro; a igualdade exata continua obrigatória em caso de colisão.
3. Introduzir gerações por categoria de registradores e por user-data. Se vertex attributes, imagens, buffers e samplers não mudaram, reutilizar o `ResolveResourceList` já materializado. Separar um cabeçalho hot (hash, gerações, ponteiros/índices compactos) dos vetores cold de especialização.
4. Para `GraphicsPipelineKey`, parar de zerar/reconstruir o objeto inteiro a cada draw quando nenhuma categoria relevante mudou. Atualizar um campo e o hash somente quando a geração daquela categoria for incrementada; manter `memcmp` como validação final.
5. Evitar buscar módulos com igualdade profunda em todos os candidatos. Um índice por fingerprint de especialização reduz a lista candidata; a comparação de `small_vector` permanece como fallback.

#### Geração correta

`WriteGraphicsRegisters` deve retornar uma máscara de categorias alteradas (shader, vertex input, raster, viewport, blend, descriptors). Uma geração global única causa invalidações excessivas; gerações separadas reduzem trabalho sem permitir estado obsoleto. Qualquer mudança que afete shader ou layout deve invalidar explicitamente todos os consumidores correspondentes.

#### Gates

Instrumentar `stage_fast_hit/miss`, motivo da invalidação, `resource_resolve_reuse`, `specialization_fingerprint_collision` e bytes comparados. Confirmar que o número de pipelines e os hashes de renderização permanecem idênticos. Não aceitar uma cache que dependa apenas de hash.

### 3. Estado dinâmico por geração — prioridade P1

`UpdateDynamicState` chama sempre cinco atualizadores. `UpdateViewportScissorState` cria vetores e percorre `NUM_VIEWPORTS` (até 16) em todo draw; `PsColorBuffer` e `PsInput` também aparecem porque a chave é reconstruída em paralelo.

Implementar dirty bits/gerações por grupo:

- viewport/scissor só é recalculado quando os registradores de viewport, scissor ou depth-range mudarem;
- depth/stencil, primitive, raster e blend usam suas próprias gerações;
- o resultado calculado fica em arrays compactos reutilizados, e `DynamicState::Commit` só recebe uma alteração quando a geração mudou;
- `ranges::equal` continua como fallback quando o grupo foi marcado dirty; não deve ser executado no caminho limpo.

Antes de usar `RenderState::operator==`, corrigir a inicialização em `BeginRendering`: `RenderState state{};` e preencher explicitamente todos os campos comparados por `memcmp`. O estado atual usa `RenderState state;`, portanto comparar bytes não inicializados mistura custo com comportamento indefinido e pode invalidar qualquer cache de renderização.

Gates: `viewport_recompute`, `dynamic_group_dirty`, `render_state_cache_hit/miss` e validação com sanitizers/validation layers quando disponíveis. O caminho limpo deve executar zero cálculos de viewport por draw.

### 4. Texture/image cache, LRU e barreiras — prioridade P1/P2

#### Cache de imagens

`TextureCache::FindImage` adquire `mutex`, calcula `ExactImageCacheKey`, consulta o mapa e chama `TouchImage`; `TouchImage` altera ponteiros da LRU mesmo quando a imagem já é o item mais recente. O perfil mostra 87,4 M chamadas a `TouchImage` e 50,3 M a `Image::GetBarriers` no novo raw.

Estratégias:

- fastpath por owner/thread para a imagem usada no draw anterior, validando UID, epoch/topology e geração de backing;
- tocar a LRU somente quando o tick avançar de fato; agrupar reordenações até o boundary de submit/GC;
- manter um índice direto por endereço/UID para hits comuns e deixar o scan de sobreposição como cold path;
- trocar invalidação global de topology por epochs dirigidos aos recursos afetados, sem remover a checagem de aliasing.

#### Barreiras

`Image::GetBarriers` cria/varre estado por mip/layer e pode emitir muitas barreiras pequenas. Manter o fastpath de no-op para layout/access iguais, mesclar intervalos adjacentes com o mesmo estado e acumular barreiras até o boundary seguro do batch. A fusão só é permitida quando origem, destino, subrecursos e ordem Vulkan são equivalentes; remover uma barreira “porque parece redundante” não é uma otimização aceitável.

Gates: `find_image_owner_hit`, `lru_reorder`, `barrier_noop`, `barrier_ranges_merged`, `topology_epoch_reject`. Conferir validation layers, aliasing e hashes de imagem antes/depois.

### 5. Decode PM4 e acesso a bitfields — prioridade P2

`PM4Type3Header::NumWords`, getters de opcode/bitfields e `NextPacket` aparecem repetidamente. A solução é reduzir o número de leituras, não escrever SIMD para `BitField` (os métodos já são `constexpr inline`):

- ler o header uma vez por pacote e materializar `opcode`/`count` em variáveis locais;
- passar o `count` ao caminho de erro e ao handler, evitando extrair o mesmo campo novamente;
- usar uma tabela estática de handlers apenas se um perfil posterior mostrar que o `switch`/branch é o custo dominante;
- manter o decode de pacotes raros em cold path e não misturar seus dados com o loop de draws.

O `ProcessCommands` também bloqueia `submit_mutex`, acessa `std::queue<UniqueFunction>` e invoca callbacks. Uma fila SPSC/MPSC tipada por owner, com storage contíguo e contadores locais, pode eliminar locks no caminho de producer único; primeiro confirmar a topologia real de produtores e preservar a ordem global exigida por `SubmitDone`.

Gates: `pm4_header_decodes`, `handler_dispatch`, `command_queue_lock_ns`, sequência de opcode/IB e nenhuma alteração nos limites de pacote.

### 6. Stream copies e memória — prioridade P2

`ExecuteStreamCopyBatch` e `CopyMappedBytes` são quentes, mas já possuem trabalho importante: fastpath para uma requisição, coalescência, cache de planos esparsos, AVX2 e cópia non-temporal para blocos grandes. Portanto, a estratégia deve ser orientada por distribuição de tamanhos, não uma reescrita SIMD genérica.

1. Instrumentar buckets de tamanho, alinhamento, número de requests e quantidade de runs mapeados; comparar bytes por segundo e impacto no frametime.
2. Adicionar caminhos especializados para 2/4 requests e runs contíguos, reutilizando um plano já calculado. Só usar hash/deduplicação quando o número de requests justificar o custo.
3. Ajustar empiricamente os limiares AVX2/non-temporal. Para cópias pequenas, `memcpy` da CRT pode vencer por usar ERMS; para grandes cópias one-shot, manter non-temporal e um único `sfence` no final do batch.
4. Em `CopySparseMemoryBatch`, preservar o cache de 64 planos e o limite de quatro runs; não aumentar estruturas hot sem medir o efeito no cache L1/L2.

Gates: histograma de cópias, `copy_plan_hit/miss`, bytes/run, `sfence_count`, tempo de `CopyMappedBytes` e frametime. SIMD só entra se produzir ganho end-to-end repetível.

### 7. Timeline, pending operations, DMA e downloads — prioridade P2

`Scheduler::PopPendingOperations` é chamado a partir de caminhos de draw/submit e atualiza `MasterSemaphore`; `CurrentTick` aparece 36,3 M vezes. A solução deve agrupar refresh/poll por submit e despachar `CompletionRecord` prontos, sem esperar ou fazer `Finish` por imagem. O download de imagens deve ser acumulado até um boundary seguro.

No `BindResources`, o caso `uses_dma` percorre todos os `mapped_ranges` sob mutex. Manter uma estrutura de intervalos dirty/registrados por página ou geração de DMA permite consultar somente os intervalos tocados pelo draw. Para fault tracking, usar uma lista compacta de páginas/bitset por chunk com deduplicação; não fazer scan completo de todas as páginas a cada evento.

Gates: `timeline_refresh_batch`, `pending_dispatch`, `download_finish_count`, `dma_ranges_scanned`, `fault_pages_unique` e correlação com gaps de idle da GPU.

## Layout de dados recomendado

As estruturas hot devem ser pequenas e contíguas:

- `QueueHot`: índices, `ready/blocked` bits, contadores e handle do submit atual; `QueueCold`: mutex, vetores DCB/CCB, callbacks e diagnósticos;
- `WaitToken`: endereço/chave, máscara, referência, função, epoch e lista compacta de owners;
- `StageOptimizationEntry` hot: fingerprints, gerações, índices e resultado; especializações e vetores de recursos em memória cold;
- estados dinâmicos em arrays SoA por grupo, com uma geração por grupo;
- barreiras e cópias acumuladas em buffers contíguos por batch, não em alocações pequenas por draw.

Isso melhora localidade L1/L2 e permite batching. Não há evidência para pinning rígido ou particionamento NUMA no Ryzen 5 5600; preservar afinidade natural do owner é mais seguro.

## Ordem de implementação sugerida

1. Corrigir `RenderState state{}` e adicionar gerações/dirty bits sem mudar semântica.
2. Instrumentar e implementar o predicado decodificado de `WaitRegMem`; comparar avaliações por pacote.
3. Introduzir ready/blocked masks e tokens com epoch, primeiro mantendo o loop antigo como fallback.
4. Otimizar o warm path de `StageOptimizationSet`, `ResolveResourceList` e `GraphicsPipelineKey` com fingerprints + igualdade exata.
5. Adicionar caches por geração para dynamic state, descriptor bindings e `FindImage`; depois mesclar barreiras seguras.
6. Amortizar timeline/download/DMA e medir `PopPendingOperations` por submit.
7. Só então ajustar thresholds de stream copy/AVX2 e, se o perfil ainda justificar, testar dispatch PM4 e SIMD específicos.

Cada fase deve ser um commit reversível. O teste é manual, mas cada run deve registrar: mesma cena e duração, PM4/draws, frametime, gaps de idle, contadores da fase e presença/ausência de artefatos visuais. O critério de sucesso é redução do p95/p99 e do tempo de frame final, não apenas menos chamadas internas.

## O que não fazer

- Não aplicar `always_inline` em todo getter de bitfield, nem SIMD manual em comparações pequenas sem medir ciclos e cache misses.
- Não remover barreiras, `SubmitDone`, epochs ou waits apenas para diminuir contadores.
- Não usar multidraw cego, timeline global única, pinning NUMA rígido ou remoção global do topology epoch.
- Não misturar raw de runs com logging e sem logging no mesmo perfil de uso do compilador; isso enviesaria o hotpath para o custo diagnóstico.

## Próxima coleta PGO

Para otimização guiada pelo compilador, usar o novo raw sem logging isoladamente e repetir a mesma cena em várias janelas equivalentes; mesclar somente esses raws. O CSV com logging deve continuar separado para provar causalidade dos tokens, filas e caches. Normalize comparações por PM4/draws, porque contagens brutas entre runs de durações diferentes enganam.

LLVM informa frequência de execução, não branch-miss ou ciclos. Depois de aplicar as primeiras fases, a confirmação de gargalo deve combinar o raw sem logging com os contadores gateados e, se necessário, um sampler externo/ETW/AMD uProf. Só após essa confirmação vale escolher inline, layout, batching ou SIMD para uma função específica.
