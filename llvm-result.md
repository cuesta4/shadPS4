# Resultado dos perfis LLVM — GOW3

## Objetivo

Este relatório analisa de forma independente os dois perfis em `Build/x64-Clang-Ultra/profiles/raw` e cruza as frequências observadas com a implementação atual. Não foram usados os dumps de telemetria do emulador nem o plano de `fast.md`.

Perfis analisados:

| Run | Arquivo | Logging do emulador | Horário |
|---|---|---|---|
| 1 | `shadps4-16532-10048391627949475423_0.profraw` | ativo | 2026-08-02 00:12:42 |
| 2 | `shadps4-31412-10048391627949475423_0.profraw` | desativado | 2026-08-02 00:39:57 |

Os dois arquivos têm o mesmo identificador de módulo, 68.951 funções, 361.055 blocos e exatamente o mesmo tamanho. Portanto, pertencem ao mesmo binário instrumentado e podem ser comparados estruturalmente.

## Como interpretar os números

O segundo perfil contém aproximadamente 1,907 vezes mais pacotes PM4 e 1,899 vezes mais draws. As runs não tiveram exatamente a mesma duração, portanto contagens absolutas não representam regressão ou ganho. Os valores foram normalizados por pacote, draw ou ocorrência do comando correspondente.

O `.profraw` de instrumentação por fonte registra quantas vezes funções e blocos foram executados. Ele não mede ciclos, cache misses, branch misses ou tempo gasto dentro do driver Vulkan. O ranking abaixo estima custo final removível combinando:

- frequência normalizada;
- trabalho executado no corpo da função;
- chamadas externas/atômicas/locks escondidas atrás do contador;
- posição no caminho crítico que alimenta a GPU;
- possibilidade de evitar o trabalho completamente, em vez de apenas torná-lo alguns ciclos mais rápido.

Além disso, o binário instrumentado usa atualização atômica dos contadores LLVM. Ele serve para descobrir fluxo e frequências, não para comparar FPS com uma build final sem instrumentação.

## Comparação normalizada

| Métrica | Logging ativo | Logging desativado | Resultado |
|---|---:|---:|---|
| Iterações de pacote gráfico | 51.936.151 | 99.039.973 | escala da run: 1,907× |
| `Rasterizer::Draw` | 1.905.763 | 3.618.565 | escala da run: 1,899× |
| Pacotes gráficos `WaitRegMem` | 12.334 | 21.220 | ocorrência do comando |
| `PM4CmdWaitRegMem::Test` | 20.358.614 | 47.531.614 | avaliações do predicado |
| Testes por `WaitRegMem` | 1.650,6 | 2.239,9 | aumento de 35,7% |
| `Task::yield_value` | 20.346.280 | 47.510.394 | praticamente um yield por teste falho |
| `PipelineCache::GetProgram` | 3.071.280 | 5.867.051 | 1,61–1,62 por draw |
| `PipelineCache::GetProgramSlow` | 2.461.902 | 4.709.069 | 80,16–80,26% das chamadas |
| Candidatos percorridos no stage cache | 8.275.228 | 15.660.944 | cerca de 6,35 por consulta cacheável |
| `ResolveStageResources` por draw | 3,523 | 3,522 | perfeitamente estável |
| `PsColorBuffer::operator==` por draw | 75,07 | 78,97 | aumento de 5,2% |
| `PsInput::operator==` por draw | 53,17 | 55,94 | aumento de 5,2% |
| `ImageSpecialization::operator==` por draw | 21,75 | 22,17 | aumento de 1,9% |
| `CompMapping::operator==` por draw | 19,94 | 20,67 | aumento de 3,7% |
| `FindImage` — hit direto | 4.372.267/4.748.714 | 8.391.361/9.091.321 | 92,07% e 92,30% |
| `TouchImage` por draw | 23,82 | 24,14 | estável e muito frequente |
| `GetBarriers` por draw | 3,98 | 4,03 | estável |
| `UpdateViewportScissorState` | 1 por draw | 1 por draw | sempre recalculado |
| Iterações de viewport | 16 por draw | 16 por draw | todos os slots sempre visitados |
| `PopPendingOperations` por draw | 1,008 | 1,007 | consulta praticamente obrigatória |
| `MasterSemaphore::CurrentTick` por draw | 9,98 | 10,03 | dez loads atômicos por draw |
| `CopyMappedBytes` por draw | 4,33 | 4,36 | estável |

Os hotpaths reais permanecem quase na mesma proporção entre os dumps. Logo, eles não foram criados pelo logging. O logging ativo adiciona, por sua vez, `AddSingleWriter` com 222,2 milhões de execuções máximas de bloco, `GetThreadRing` com 146,2 milhões e coleta de timestamps/histogramas. Essas operações desaparecem do topo com logging desligado.

## Ranking por ganho final esperado

### 1. `WaitRegMem` + retomada de coroutines/filas

Arquivos: `src/video_core/amdgpu/liverpool.cpp:205`, `src/video_core/amdgpu/liverpool.cpp:1007` e `src/video_core/amdgpu/pm4_cmds.h:684`.

Este é o maior desperdício comprovado no caminho crítico. Na run sem logging, somente 21.220 comandos causaram 47,53 milhões de avaliações, 47,51 milhões de yields e aproximadamente o mesmo número de passagens por `Liverpool::Process`. Cada falha volta ao escalonador, consulta comandos, avança a fila, adquire `queue.m_access`, retoma a coroutine e lê novamente endereço, máscara, função e memória.

O custo agregado não está apenas em `Test`; é a cadeia inteira disparada por cada poll:

`WaitRegMem::Test → yield → Liverpool::Process → scan/lock/resume → Test`

#### Otimização recomendada

1. Decodificar `mem_space`, função, máscara, referência e endereço/registro uma única vez ao entrar no pacote. O loop não deve repetir extrações de `BitField` nem reconstruir o endereço.
2. Criar fastpaths exatos para os predicados comuns: memória + `Equal`/`NotEqual`, com máscara cheia ou parcial. Os demais casos continuam usando o switch atual.
3. Substituir polling contínuo por um `WaitToken` registrado por endereço/página e geração. Escritas que possam satisfazer a espera marcam o token como pronto e colocam sua fila no `ready_mask`.
4. Separar `ready_mask` e `blocked_mask` em `u64`. O owner escolhe uma fila pronta com `std::countr_zero`, sem varrer filas bloqueadas.
5. Manter o loop atual como fallback para endereços que não possam ser observados com segurança. O token é uma otimização; a leitura real da memória continua sendo a confirmação final.

#### Layout orientado a dados

`WaitToken` deve conter apenas chave/endereço, máscara, referência, função, epoch e queue id. Dados de diagnóstico, mutexes e callbacks ficam numa estrutura cold. Filas prontas devem armazenar handles contíguos; `std::queue` e mutex por resume não devem estar no caminho owner-only.

#### Validação

Medir `wait_packets`, `wait_predicate_evals`, `wait_token_hit/miss`, `stale_epoch`, `queue_scans` e `queue_resumes`. O critério é reduzir milhares de avaliações por wait para aproximadamente uma avaliação por wakeup, sem avanço prematuro de `rptr`, lost wakeup ou mudança de ordem PM4.

### 2. `Scheduler::PopPendingOperations` e `MasterSemaphore::Refresh`

Arquivos: `src/video_core/renderer_vulkan/vk_rasterizer.cpp:218`, `src/video_core/renderer_vulkan/vk_scheduler.cpp:122` e `src/video_core/renderer_vulkan/vk_master_semaphore.cpp:86`.

`PopPendingOperations` é chamado praticamente uma vez por draw. A função sempre:

1. adquire `priority_pending_ops_mutex`;
2. chama `MasterSemaphore::Refresh`;
3. executa `vkGetSemaphoreCounterValue` através de Vulkan;
4. só depois verifica se `pending_ops` contém algo pronto.

O perfil LLVM não instrumenta o driver Vulkan, portanto a chamada externa aparece como uma única execução embora possa custar muito mais que dezenas de getters C++. A fila esteve no corpo de processamento apenas 89.252 vezes no primeiro perfil e 85.223 no segundo, contra 1,92 e 3,64 milhões de chamadas: a vasta maioria das consultas não despachou nada.

#### Otimização recomendada

- manter um contador/flag atômico `pending_ops_count` atualizado quando uma operação é adicionada ou removida;
- retornar imediatamente quando o contador for zero, sem mutex e sem consultar a timeline;
- quando houver itens, comparar primeiro o front com `KnownGpuTick`; chamar `Refresh` apenas se o tick conhecido ainda não for suficiente;
- separar o mutex de `pending_ops` do mutex da fila prioritária, ou tornar `pending_ops` owner-only se todos os produtores reais estiverem no scheduler;
- despachar todas as operações liberadas após um único refresh;
- atualizar o tick conhecido no boundary de submit/completion e reutilizar o snapshot durante o draw.

Essa mudança troca milhões de locks e chamadas ao driver por um load relaxed no caminho vazio. Deve ser uma das primeiras implementações por combinar alto potencial com escopo pequeno.

### 3. `PipelineCache::GetProgram`, especializações e recursos resolvidos

Arquivos: `src/video_core/renderer_vulkan/vk_pipeline_cache.cpp:63`, `:554`, `:982` e `:1069`.

O stage cache atual não está cumprindo sua função no warm path:

- `GetProgram` invalida `cache_set.current` no início de toda chamada;
- cerca de 80% das chamadas chegam a `GetProgramSlow`;
- nas consultas cacheáveis são visitados em média 6,35 dos 8 slots;
- antes da busca, `RefreshDynamicProgramData` e todos os `ResolveResourceList` são executados;
- o slow path constrói `StageSpecialization` e faz busca linear nos módulos.

Isso explica os maiores contadores reais da run sem logging: 285,8 M em `PsColorBuffer::operator==`, 202,4 M em `PsInput::operator==`, 80,2 M em `ImageSpecialization::operator==`, 74,8 M em `CompMapping::operator==` e 61,3 M em `VsAttribSpecialization::operator==`.

#### Otimização recomendada

1. Validar primeiro `cache_set.current` em vez de apagá-lo. GOW3 apresenta forte localidade temporal; o último stage usado deve ser o primeiro candidato.
2. Associar ao slot um cabeçalho hot: `program_hash`, `program_base`, logical stage, geração de user-data, revisão do fetch shader, fingerprint de runtime e índice do resultado.
3. Só materializar `ResolvedStageResources` quando uma geração relevante mudou. Hoje são feitas 3,522 resoluções de stage por draw, mesmo em pipeline quente.
4. Usar fingerprint para localizar candidatos, seguido obrigatoriamente de comparação exata. Hash nunca pode ser a única prova de igualdade.
5. Indexar `Program::modules` por fingerprint de `StageSpecialization`; manter a busca linear apenas para colisões/fallback.
6. Separar o cabeçalho do cache e os arrays de recursos/especializações. O loop quente deve percorrer dados compactos que caibam em poucas cache lines.
7. Para `GetGraphicsPipeline`, guardar a dependency key já validada por gerações. Não reconstruir recursos e executar `EqualResolvedResources` se nenhuma geração consumida pelo stage mudou.

#### Resultado esperado

O fastpath ideal executa uma comparação de gerações, valida o slot atual e retorna. Não deve construir `RuntimeInfo`, resolver descriptors nem percorrer módulos quando o estado relevante é idêntico ao draw anterior.

### 4. `BindPipelineResources` e construção de descriptors

Arquivo: `src/video_core/renderer_vulkan/vk_rasterizer.cpp:432` e `:508`.

`BindPipelineResources` foi chamado 3,63 milhões de vezes na run sem logging e percorreu aproximadamente 7,38 writes por chamada. O cache atual compara somente `VkDescriptorImageInfo`; writes de buffers e outros descriptors continuam sendo reconstruídos e enviados. `BindResources` também limpa e repopula vários vetores em todo draw.

#### Otimização recomendada

- criar um `DescriptorBindingPlan` por pipeline, contendo offsets/índices fixos e apenas campos dinâmicos;
- versionar buffers, imagens, samplers e user-data separadamente;
- armazenar snapshot completo de buffer handle/offset/range, image view/layout e sampler;
- gerar somente writes cujos campos mudaram;
- pular a chamada de push descriptors quando não houver write nem barrier, preservando separadamente push constants/user-data que realmente mudaram;
- reservar storage fixo contíguo e reutilizá-lo, evitando clear/resize/reconstrução dos vetores a cada draw.

O gate é contar writes produzidos, writes evitados e chamadas Vulkan efetivamente emitidas. Uma taxa alta de cache hit sem redução das chamadas ao driver não é suficiente.

### 5. Estado dinâmico e `WriteGraphicsRegisters`

Arquivos: `src/video_core/amdgpu/liverpool.cpp:360` e `src/video_core/renderer_vulkan/vk_rasterizer.cpp:1409`.

`WriteGraphicsRegisters` alcança 73,34 milhões de execuções na run sem logging. Para cada write, a função pode ler o intervalo com `memcmp`, percorrer a máscara de registradores de pipeline e depois ler novamente para `memcpy`. Ao mesmo tempo, `UpdateDynamicState` recalcula todos os grupos em todo draw. `UpdateViewportScissorState` visita exatamente 16 viewports por draw, produzindo 57,90 milhões de iterações.

#### Otimização recomendada

1. Fazer compare-and-copy especializado para os tamanhos comuns de 1, 2, 4 e 8 dwords. Para blocos pequenos, uma passagem deve copiar e produzir a máscara de mudança; evitar `memcmp` + segundo scan + `memcpy`.
2. Dividir `graphics_pipeline_generation` em gerações por categoria: shader, resources, vertex input, viewport/scissor, depth/stencil, raster e blend.
3. Em `UpdateDynamicState`, retornar imediatamente para cada grupo cuja geração não mudou. O resultado calculado de viewport/scissor fica cacheado em arrays fixos.
4. Recalcular apenas os viewports ativos e somente quando registradores relevantes mudarem. O caminho limpo deve executar zero iterações de viewport.
5. Propagar as mesmas gerações para pipeline e descriptor caches, evitando mecanismos paralelos de invalidação.

Antes de depender de igualdade byte a byte no render-state cache, `RenderState state` em `vk_rasterizer.cpp:1096` deve ser inicializado como `RenderState state{}`. `RenderState::operator==` usa `memcmp`; bytes não inicializados tornam o cache incorreto e não determinístico.

### 6. `TextureCache::FindImage`, `TouchImage` e `Image::GetBarriers`

Arquivos: `src/video_core/texture_cache/texture_cache.cpp:547`, `:1184`, `src/common/lru_cache.h:29` e `src/video_core/texture_cache/image.cpp:221`.

O cache direto de imagens funciona: cerca de 92% das chamadas de `FindImage` retornam pelo exact cache em ambos os dumps. Entretanto, até o hit adquire o mutex global, calcula/compara a chave, lê `CurrentTick` e toca a lista LRU. No agregado há 87,36 milhões de `TouchImage` e 52,61 milhões de `GetImage` na run sem logging.

#### Otimização recomendada

- fastpath owner-only antes do mutex usando `(ImageId, UID, topology epoch, backing generation)`;
- só entrar no caminho de overlap/mutex quando o token falhar;
- capturar o scheduler tick uma vez por draw/submit e passá-lo às operações de cache;
- acumular touches em um bitset/lista compacta e atualizar a LRU uma vez por submit ou GC;
- manter uma marca por imagem para não tocar o mesmo item dezenas de vezes no mesmo tick;
- dividir invalidação global de topology em gerações dirigidas ao backing/alias afetado, mantendo o epoch global como fallback de segurança.

`Image::GetBarriers` é chamado cerca de quatro vezes por draw. Quando há estado parcial, ele percorre mip × layer e pode emitir uma barreira por subrecurso. Intervalos adjacentes com layout/access/stage idênticos podem ser fundidos numa única `VkImageMemoryBarrier2`; transições no-op devem retornar antes de materializar o vetor. A fusão deve preservar exatamente ordem, access masks e ranges Vulkan.

### 7. Stream copies e `CopyMappedBytes`

Arquivos: `src/video_core/buffer_cache/buffer_cache.cpp:238` e `src/core/memory.cpp:55`/`:263`.

Na run sem logging, `ExecuteStreamCopyBatch` foi chamado 3,60 milhões de vezes, sua busca/coalescência interna atingiu 63,40 milhões de iterações e `CopyMappedBytes` foi chamado 15,77 milhões de vezes. Este é trabalho de memória real, mas a implementação já contém:

- fastpath para uma request;
- deduplicação linear para lotes pequenos e hash para lotes maiores;
- cache thread-local de planos esparsos;
- AVX2 para até 256 bytes;
- stores non-temporal para blocos grandes alinhados.

#### Otimização recomendada

Primeiro coletar histograma de tamanho, alinhamento, quantidade de requests, runs e taxa de deduplicação. Depois:

- especializar lotes de 2 e 4 requests sem tabela hash;
- fundir requests guest contíguas antes de `CopySparseMemoryBatch`;
- calcular `source_key` uma vez e armazená-la em scratch SoA;
- ajustar os thresholds AVX2/non-temporal com benchmark do hardware-alvo;
- emitir um único `sfence` por batch;
- manter `memcpy` para classes em que a CRT/ERMS vencer.

SIMD adicional só deve ser implementado se reduzir ciclos/byte e frametime. Os counters atuais provam volume, mas não dizem se o limite é bandwidth, cache ou overhead por request.

### 8. Telemetria quando logging está desligado

Mesmo com logging desativado, `Log::IsEnabled` e `PerformanceTelemetry::Enabled` atingem 305,6 milhões de blocos, `PerformanceTelemetry::Add` 151,4 milhões e os construtores/destrutores de `ScopedDuration` 54,95 milhões. Não há escrita no ring, mas ainda existem loads/branches e objetos RAII no caminho quente.

#### Otimização recomendada

- capturar `telemetry_enabled` uma vez por frame/submit ou por iteração externa do GCP;
- chamar variantes `AddEnabled` somente dentro desse bloco;
- permitir que a GUI atualize um epoch global; cada thread renova seu snapshot no próximo boundary seguro;
- não consultar `Log::IsEnabled` para cada descriptor, pacote, viewport ou resume.

Assim a opção continua dinâmica, mas o custo desligado cai de centenas de milhões de checks para poucas verificações por frame/submit. Para gerar o perfil de produção do compilador, usar somente raws com logging desligado.

### 9. `LibAtrac9::Dct4`

Arquivo: `externals/LibAtrac9/C/src/imdct.c:25`.

`Dct4` teve 58.166 chamadas/26,06 M iterações no primeiro dump e 115.683 chamadas/51,83 M iterações no segundo. É um custo recorrente real. O corpo executa butterflies escalares em `double`, e o relatório de otimização LLVM indica que os loops não foram vetorizados no binário instrumentado.

O caminho pode ser reescrito em blocos AVX2 de quatro doubles, com arrays alinhados, ponteiros `restrict` e kernels especializados pelo `MdctBits`. Outra alternativa é portar uma IMDCT vetorizada já validada bit-a-bit para ATRAC9. A própria instrumentação atômica pode impedir a vetorização; antes de escrever intrinsics, é necessário conferir o assembly/optimization record da build PGO final sem counters. O teste precisa comparar PCM produzido e verificar drift/ruído.

Apesar do potencial SIMD, este trabalho ocorre na thread de áudio. Ele só melhora os 60 FPS se competir por CPU/cache ou bloquear o produtor gráfico; por isso fica abaixo dos caminhos GCP/Vulkan serializados.

## Custos que não devem orientar o trabalho de 60 FPS

`ImFontAtlas::GetTexDataAsRGBA32`, `stbtt__h_prefilter` e `stbtt__rasterize_sorted_edges` têm contadores internos altos, mas são idênticos nos dois dumps e `GetTexDataAsRGBA32` foi chamado uma única vez em cada run. São custo de inicialização do atlas, não custo por frame. Otimizá-los não melhora a cena em execução.

Os getters de `BitField` também não devem ser “otimizados” isoladamente com `always_inline`: já são triviais e inline. O ganho vem de não extrair o mesmo campo milhões de vezes dentro de `WaitRegMem` e do decode PM4.

## Ordem recomendada de implementação

1. Fastpath vazio de `PopPendingOperations`, evitando lock e `vkGetSemaphoreCounterValue` sem trabalho pendente.
2. Predicado pré-decodificado de `WaitRegMem`, seguido de `WaitToken` + ready/blocked masks.
3. Fastpath do slot atual em `GetProgram` e cache por gerações de recursos/runtime.
4. Gerações por categoria em `WriteGraphicsRegisters`; usar as gerações para pular dynamic state e pipeline work.
5. Cache completo de descriptors e redução efetiva das chamadas Vulkan.
6. Token owner-only para `FindImage`, batching da LRU e fusão segura de barreiras.
7. Histogramas e especializações para stream copies; somente depois ajustar SIMD.
8. Coarsening do gate de telemetria desligada.
9. AVX2 para `Dct4` se um sampler confirmar impacto relevante fora da thread de áudio.

As fases 1 e 2 atacam trabalho serializado e chamadas externas/polls; são as mais prováveis de reduzir diretamente p95/p99. As fases 3 a 6 removem trabalho constante de cada draw e são as mais prováveis de recuperar milissegundos de média.

## Uso dos perfis pelo compilador

Para construir a versão PGO voltada a jogar sem logging, o perfil correto é o segundo raw, ou uma mescla de novas runs equivalentes também sem logging. Misturar o primeiro perfil faria LLVM gastar orçamento de inline/layout em `AddSingleWriter`, ring buffers, timestamps e histogramas que não existem no uso desejado.

PGO pode melhorar layout, inlining e previsão de branches automaticamente, mas não eliminará as operações arquiteturais acima: polling de coroutine, chamada Vulkan por draw, resolução repetida de recursos, mutex global e reconstrução de descriptors. Esses exigem mudanças de código.

## Limite de certeza e confirmação necessária

O ranking identifica onde existe maior trabalho agregado removível e quais chamadas escondem custo externo. Ele não atribui milissegundos porque `.profraw` não contém amostras de CPU/PMU. Após cada implementação, a confirmação deve usar a mesma cena, build final sem instrumentação LLVM e medir frametime p50/p95/p99. Para desempatar dois hotpaths restantes, usar ETW/AMD uProf ou timers seletivos gateados; não usar o FPS da build instrumentada como resultado final.
