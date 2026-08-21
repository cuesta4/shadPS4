# Findings auditados — queda de performance ao mover câmera/Kratos

Data da auditoria: 2026-08-15  
Branch: `codex/performance-opt` @ `571b3e0dc5c773c7adc587e5fa1569a393be2ef2` (`dirty`)  
Jogo: God of War III Remastered, CUSA01715, versão 1.02, SDK guest 2.00  
Host: Ryzen 5 5600, RTX 3060, Vulkan NVIDIA 610.88

Dump-base v7 lido integralmente:

- arquivo: `C:/Users/Arthur/AppData/Roaming/shadPS4/log/shadps4-telemetry-1786804216034.csv`;
- schema: v7;
- tamanho: 48.694.599 bytes;
- SHA-256: `0604B61AFAC5F13FE1CCC189675EF4A407393E4F41E805FE4C1183630DE5D573`;
- 770.099 linhas incluindo o header, 770.098 registros válidos e zero linhas malformadas;
- duração declarada: 53,5365884 s;
- 17 threads;
- 2.600 registros de frame, sem overwrite;
- 27.284 registros de imagem/readback, sem overwrite;
- 246.734 eventos genéricos escritos, dos quais 228.880 retidos e 17.854 sobrescritos.

Build que gerou o dump v7: `Build/x64-Clang-Release/shadps4-perf.exe`, SHA-256
`2175EAE92FB5C0076DB603C1C441BF9661BAA019CC9825AC95F26B9FB0F4CA1B`.

Build instrumentada v8 preparada para a próxima run:
`Build/x64-Clang-Release/shadps4-perf.exe`, SHA-256
`F6CA05E75055C2DD1B491FA56E82B17ED6806B6037D6A3F57BFD09341B9B319C`.
Ela foi compilada com `ENABLE_DETAILED_TELEMETRY=ON`; o alias é uma cópia byte a byte de
`Build/x64-Clang-Release-Telemetry/shadps4.exe`.

O executável Release não instrumentado permaneceu inalterado:
`Build/x64-Clang-Release/shadps4.exe`, SHA-256
`A560901DFC4594FFD33F012D23F450B779F8032D13DC2DF29E348DA06063BB51`.

Fontes cruzadas nesta auditoria:

- CSV v7 completo e logs da mesma execução;
- código-fonte atual dos caminhos de draw, pipeline, staging, scheduler e texture cache;
- assembly/objetos Release já auditados nesta branch;
- SDK oficial PS4 4.508.001, especialmente Gnm/Gnmx, SRT, EOS, EOP e waits.

O jogo declara SDK guest 2.00, enquanto o corpus oficial disponível é 4.50. Os encodings PM4 e os
contratos usados aqui coincidem com o stream observado, então o SDK 4.50 é evidência forte do
comportamento de hardware/API; ele não prova qual versão de helper Gnmx o binário guest chamou.
Quando a intenção do jogo importa, os bytes PM4 observados permanecem a fonte primária.

`SET_PREDICATION` continua deliberadamente fora do escopo.

---

## 0. Vocabulário de evidência

Uma hipótese só recebe a marca **[ready]** quando os dados atuais são suficientes para desenhar uma
correção concreta, preservar os contratos observados e medir a remoção de trabalho. Volume alto ou
assembly grande, isoladamente, não bastam.

- **FATO:** observado diretamente no CSV completo, código, assembly ou SDK.
- **INFERÊNCIA SUPORTADA:** explicação que fecha com múltiplas observações, mas ainda precisa de uma
  medição discriminante.
- **[ready]:** mecanismo, invalidadores e gate de correção estão suficientemente definidos para uma
  implementação futura.
- **[needs-v8]:** a hipótese continua plausível, mas uma decisão de implementação ainda exigiria
  assumir algo que o dump v7 não mede.
- **REBAIXADO:** mecanismo real, porém pequeno ou sem relação suficiente com a queda desta cena.

Há duas grandezas distintas:

1. custo medido do scope atual;
2. fração realmente removível por uma correção.

As estimativas deste documento nunca excedem o primeiro valor sem declarar explicitamente que se
trata de um efeito indireto ainda não isolado.

### 0.1 Estado executivo

| Estado | Hipótese | Ganho plausível na cena final | Confiança |
|---|---|---:|---|
| **[ready]** | transient read stream: resolver dense + ring host-direct | **2,1–4,0 ms nos frames lentos** sem contar reuse; **3–5+ ms** se houver conteúdo estável | alta para o mecanismo; média-alta para o ganho até o A/B de memory type |
| **[ready]** | generations independentes por grupo de dynamic state | **0,15–0,25 ms/frame**; teto medido de **0,305 ms** nos 15 s finais | alta para o mecanismo, média-alta para o ganho |
| **[ready]** | fast path pós-resolução para o estágio SRT atualmente não cacheável | **0,01–0,02 ms/frame** diretamente plausíveis | alta para o micro-fastpath; walker/skip inseguro continua pendente |
| rebaixado | reutilização de descriptors através de troca de pipeline compatível | **0 ms/frame demonstrado**; compatibilidade isolada não autoriza reuso | alta: exact-state e reusable hits foram zero |
| rebaixado | deferral de readback/submit | **0,2–0,5 ms/frame** diretamente sustentados; não 5–12 ms | alta para o custo direto, baixa para efeito indireto |

A implementação principal pronta passa a ser staging, fechada pelo schema v9 na seção 13. Dynamic
state e o fast path SRT pós-resolve continuam prontos, mas são secundários pelo ganho medido.
Readback permanece semanticamente delicado, e os dados retiram a justificativa para tratá-lo como o
maior gargalo desta cena.

---

## 1. Como interpretar o CSV

### 1.1 Cobertura

Counters globais, frames e writebacks cobrem a sessão inteira. O overwrite de **17.854 registros**
atingiu somente o ring genérico de eventos antigos. Portanto:

- totais globais podem ser usados para toda a sessão;
- os 2.600 registros de frame podem ser usados para análise temporal completa, descartando o
  primeiro, deixando **2.599 intervalos**;
- os 27.284 registros de writeback têm cobertura completa;
- uma cronologia baseada apenas em `event` não deve presumir cobertura do início da sessão.

O primeiro `frame` acumula startup e é descartado. Restam 2.599 intervalos. Os últimos 15 segundos
são privilegiados porque o usuário os identificou como a parte relevante da cena.

### 1.2 Timers amostrados

Para `timer_sample` e `frame_timer`:

```text
tempo_estimado = value_ns_amostrado * sample_period
média_por_chamada_amostrada = value_ns_amostrado / samples
```

Scopes aninhados não são somados. Por exemplo, as fases v7 de staging estarão contidas em
`staging_stream_batch`.

O relógio observado tem granularidade próxima de 100 ns. Timers muito curtos são úteis em agregado,
mas médias ingênuas podem ser contaminadas por preempção. Por isso foram cruzados:

- soma extrapolada;
- histograma p50/p90/p95/p99;
- estabilidade da média por chamada entre frames rápidos e lentos;
- fechamento contra counters exatos.

`descriptor_prepare` é o principal exemplo de cauda contaminada: p50 de aproximadamente 200 ns,
mas poucos outliers de até milissegundos inflam sua média. O total ingênuo não é tratado como CPU
removível sem a decomposição v7.

### 1.3 Build detailed versus Release

A build detailed faz contagem PM4, timers e amostras em hot paths. Seu FPS absoluto não é comparável
à build Release. São comparações válidas:

- detailed A versus detailed B, mesma cena;
- Release A versus Release B, mesma cena;
- relações internas e trabalho por frame dentro de uma única run detailed.

O log confirma readback linear ativo, FSR ativo, validation layers desligadas e pipeline cache ativo.
Não houve compilação recorrente de pipelines nesta run.

---

## 2. O que caracteriza a queda ao mover

### 2.1 Frame pacing

Após descartar o primeiro frame:

| Janela | N | média | p50 | p95 | frames `>=30 ms` |
|---|---:|---:|---:|---:|---:|
| sessão | 2.599 | 19,887 ms | 16,670 ms | 34,464 ms | 17,70% |
| 15 s anteriores aos finais | 706 | 21,246 ms | 16,710 ms | 34,540 ms | 27,48% |
| 15 s finais | 734 | 20,437 ms | 16,675 ms | 34,670 ms | 22,62% |

A distribuição continua bimodal em aproximadamente 16,7/34 ms. Um pequeno estouro do budget pode
perder um vblank e aparecer como aproximadamente 17 ms adicionais; esses 17 ms não precisam ser
17 ms de novo trabalho CPU.

### 2.2 O último trecho tem menos trabalho, mas os frames lentos continuam sendo workload bursts

Últimos 15 s contra os 15 s anteriores:

- draws/frame: 2.082 contra 2.154, **−3,4%**;
- `draw_cpu_ns`/frame: 10,830 contra 11,253 ms, **−3,8%**;
- PM4/frame: 47.843 contra 49.228, **−2,8%**;
- staging/frame: 24,423 contra 26,700 MB, **−8,5%**;
- `staging_stream_batch`/frame: 5,217 contra 5,367 ms, **−2,8%**;
- slow path de estágio/frame: 0,598 contra 0,628 ms, **−4,8%**;
- dynamic state/frame: 0,305 contra 0,316 ms, **−3,6%**.

Dentro dos 15 s finais, a distribuição continua bimodal:

- 564 frames rápidos (`<=18 ms`): 16,436 ms em média, 1.777 draws/frame,
  **5.294 µs/draw** e 19,535 MB de staging/frame;
- 166 frames lentos (`>=30 ms`): 34,088 ms em média, 3.121 draws/frame,
  **5.026 µs/draw** e 41,030 MB de staging/frame;
- o `staging_stream_batch` estimado foi 4,436 ms/frame nos rápidos e 7,889 ms/frame nos lentos.

**Conclusão suportada:** nesta run, a janela final não piorou globalmente; ela melhorou. Ainda assim,
os frames lentos executam aproximadamente 76% mais draws, 110% mais staging e quase o dobro do
custo agregado do batch que os rápidos, enquanto o custo por draw não aumenta. A queda continua
sendo uma amplificação episódica de workload, não uma mudança de modo por chamada. Isso direciona
a otimização para trabalho por draw/staging e para reduzir bursts que cruzam o vblank.

Essa conclusão não afirma que todo aumento seja inevitável. Pelo contrário: ela torna especialmente
valiosos fast paths de poucas centenas de nanos executados milhares de vezes por frame.

---

## 3. Staging e stream copies — maior custo ainda não decomposto

### 3.1 Fatos do v7

- `staging_bytes`: 53.213.570.476 bytes;
- `staging_stream_batch`: aproximadamente **11,484 s**, ou **4,42 ms/frame** na sessão;
- nos 15 s finais: **5,22 ms/frame**;
- p50 por batch: 1.024–1.151 ns;
- p90: 6.144–6.655 ns;
- p95: 9.216–10.239 ns;
- p99: 13.312–14.335 ns;
- máximo amostrado: 61–65 µs.

O histograma é estável e a média por batch não cresce nos frames lentos. Diferentemente do total
ingênuo de descriptor prepare, este custo é robusto.

A extrapolação amostrada atribui aproximadamente:

- 4,50 milhões de batches múltiplos e aproximadamente 39,86 GB alocados no stream batch;
- 4,53 milhões de draws e 4,53 milhões de misses de dynamic state;
- 1.185.122 requests amostrados e 1.185.122 cópias canônicas;
- zero reusos exatos e zero reusos por subrange; deduplicação está refutada nesta cena;
- 1.184.981 cópias canônicas guest nos mesmos samples.

Os batches mais comuns alocam poucos KiB. Isso é compatível com overhead de muitos pequenos
requests e cache pollution; não é evidência de saturação da largura de banda DRAM.

### 3.2 O que o v7 resolveu e o que continua pendente

`BufferCache::ExecuteStreamCopyBatch` inclui no mesmo timer:

1. hash, igualdade, deduplicação exata e busca por subrange;
2. cálculo de offsets/alinhamento e `StreamBuffer::Map`;
3. construção das cópias canônicas, `CopySparseMemoryBatch`, memcpy/memset e commit;
4. materialização de um resultado por request original.

O v7 mostra que deduplicação não remove trabalho: requests e cópias canônicas coincidem, e os dois
contadores de reuse são zero. O copy phase continua dominante, mas o timer inclui tanto a preparação
das `SparseCopyRequest` quanto a própria cópia sparse, além de host/zero copies e commit.

Saber que o conjunto custa 5,22 ms/frame não revela qual transformação remove ciclos. Uma tentativa
de otimizar a estrutura inteira agora teria de adivinhar entre hash, scans, mapping, cópia real e
fan-out dos resultados.

### 3.3 Instrumentação v7

Foram adicionados timers amostrados 1/64:

- `staging_batch_deduplicate`;
- `staging_batch_layout`;
- `staging_batch_copy`;
- `staging_batch_results`.

Foi adicionada uma amostra uniforme por batch 1/16, com counters globais e por frame:

- `staging_batch_samples`;
- `staging_batch_requests`;
- `staging_batch_canonical_copies`;
- `staging_batch_guest_copies`;
- `staging_batch_exact_reuses`;
- `staging_batch_subrange_reuses`;
- `staging_batch_requested_bytes`;
- `staging_batch_canonical_bytes`;
- `staging_batch_allocated_bytes`.

As contagens de fonte do stream batch passaram a ser agregadas uma vez por batch amostrado, em vez
de chamar a instrumentação por cópia canônica. Isso mantém o estimador e reduz a interferência do
próprio diagnóstico.

### 3.4 Instrumentação v8 preparada

O schema v8 adiciona um gate uniforme 1/16 por batch para os counters de forma e para a decomposição
sparse. Os timers continuam usando o mecanismo de amostragem temporal próprio (1/64 por site),
porque precisam estimar custo de CPU sem transformar cada chamada em uma medição. Portanto, os
counters sparse e os timers devem ser extrapolados separadamente; não se deve cruzar a linha de um
timer com a linha de um counter como se fossem necessariamente o mesmo subconjunto de batches.
Foram adicionados:

- `staging_batch_request_prepare`: tempo preparando as cópias canônicas, incluindo a construção das
  `SparseCopyRequest` guest e as cópias host/zero que ocorrem no mesmo loop;
- `staging_sparse_copy`: tempo dentro de `MemoryManager::CopySparseMemoryBatch` para batches do
  stream path; chamadas singleton de `CopySparseMemory` não entram neste timer;
- `staging_sparse_copy_samples`, requests e requested/copied bytes;
- hits/misses do cache de planos, planos construídos e fallbacks `copy_reference`;
- runs e bytes mapped/zero, além de bytes que passaram pelo fallback.

Os counters sparse são coletados somente no mesmo gate 1/16 que incrementa
`staging_batch_samples`, quando o batch possui pelo menos uma cópia canônica `Guest` (batches
somente `Host`/`Zero` retornam antes do sparse path). Por isso requests, bytes, hits/misses, planos,
fallbacks e runs podem ser fechados entre si multiplicando `staging_sparse_copy_sample_period`
(16); o denominador de batches sparse é `staging_sparse_copy_samples`, não
`staging_batch_samples`. O timer
`staging_sparse_copy`, por outro lado, é uma amostra independente 1/64 do scope
`CopySparseMemoryBatch`. `staging_batch_copy` v7 permanece o timer pai e não deve ser somado às
duas subfases: a diferença inclui host/zero copies e `StreamBuffer::Commit`. A próxima run decide
se o alvo é o plano sparse, a preparação de requests ou o restante do commit, sem inferência
estrutural.

Os invariantes úteis para validar o dump são: `plan_hits + plan_misses = requests` para cada soma
sparse; `plans_built + copy_reference_fallbacks = plan_misses`; e
`copied_bytes = mapped_bytes + zero_bytes`. `reference_bytes <= copied_bytes` é apenas um marcador
de subconjunto: ele identifica os bytes que caíram no walker `copy_reference`, mas esses bytes já
estão incluídos em `mapped_bytes` ou `zero_bytes` e não podem ser somados novamente. `mapped_runs`
e `zero_runs` contam os segmentos de VMA, não os requests.
Assim, a run poderá distinguir custo de lookup/construção do plano, custo de caminhar VMAs e custo
de cópia/memset sem alterar a operação emulada.

Gate para **[ready]**:

- uma fase precisa explicar uma fração material e estável do total;
- requests/canonical e requested/canonical bytes precisam indicar se dedup/fan-out realmente paga;
- a proposta deve preservar alinhamento, offsets, source type, ausência de overread e lifetime da
  slice até o tick Vulkan correspondente;
- redução precisa aparecer na fase e no `draw_cpu_ns`, não só em uma contagem proxy.

Estimativa provisória: **1–3+ ms/frame** continua plausível; **5,22 ms/frame** é o teto medido do
scope nos 15 s finais. Nenhuma correção semântica de staging está [ready] antes do v8.

---

## 4. SRT e especialização de estágio — [ready]

### 4.1 Contrato do SDK

O SDK oficial permite que SRT contenha pointers e estruturas indiretamente referenciadas, inclusive
níveis adicionais de indireção. O tutorial oficial recomenda dados compartilhados com a CPU em
estruturas alcançadas indiretamente pela SRT. Portanto, o endereço SRT top-level e os 16 dwords de
user data podem permanecer iguais enquanto o conteúdo dereferenciado muda.

Consequência obrigatória: user-data bruta ou identidade do pointer não autoriza pular o walker SRT.
Hash também não substitui igualdade exata. Um cache anterior ao walker exige generations de todas
as páginas/faixas dereferenciadas e de todas as fontes de escrita; isso não existe hoje.

Após executar walker e `ResolveStageResources`, porém, comparar a especialização materializada com a
permutação atual é uma decisão diferente e potencialmente segura. Essa separação é a base do
candidato atual.

TCS/TES continuam excluídos de qualquer fast path simples: `StageSpecialization` também lê
constantes indiretas de tessellation e altera `RuntimeInfo`. Elas teriam de fazer parte da key ou ser
recalculadas antes do match.

### 4.2 Fatos do v7

- `stage_cache_uncacheable = stage_specialization_builds = stage_permutation_hits = 4.321.730`;
- zero nova permutação e zero compile recorrente;
- todos os reasons amostrados são exatamente `0x9`:
  `SRT walker | descriptor outside direct user data`;
- extrapolação por estágio: aproximadamente 2,19 M vertex, 2,11 M fragment e 26,8 mil compute;
- slow path total: **0,503 ms/frame** na sessão e **0,598 ms/frame** nos 15 s finais;
- construção de `StageSpecialization`: aproximadamente 0,367 ms/frame na sessão;
- a busca encontra alguma permutação existente em 100% dos casos observados;
- no resultado slow-path, `stage_slow_current_hits` é 4.199.899 (97,18%) e
  `stage_slow_other_hits` é 121.831 (2,82%).

Assembly atual continua coerente com o diagnóstico: o caminho não cacheável constrói um objeto de
especialização grande e faz busca linear. O target é `x86-64-v3`, logo AVX2 está disponível; isso
não torna seguro comparar menos dependências.

### 4.3 Ambiguidade restante

O v7 resolve a ambiguidade: a grande maioria das especializações já materializadas coincide com a
`current_permutation`. Isso autoriza um fast path pós-resolve que compara a especialização completa
com a corrente, sem pular o walker SRT. Os 2,82% de `other_hits` não justificam um índice estrutural
mais complexo sem outra medição.

### 4.4 Instrumentação v7

Foram adicionados counters exatos, também em `frame_counter`:

- `stage_slow_current_hits`;
- `stage_slow_other_hits`;
- `stage_slow_search_comparisons`.

O último conta quantas especializações a busca linear realmente examinou. `stage_permutation_compiles`
já cobre a saída sem hit.

Gate para **[ready]**:

- se current hits dominarem, executar walker/resolve e comparar exatamente a permutação corrente;
- se other hits dominarem e houver várias comparações, desenhar índice por fingerprint seguido de
  igualdade exata;
- nunca usar raw user data para declarar SRT igual;
- manter TCS/TES fora até suas dependências indiretas estarem representadas;
- validar queda de `stage_specialization_builds`, `stage_slow_path` e `draw_cpu_ns`.

Ganho plausível do micro-fastpath pós-resolve: **0,01–0,02 ms/frame**, porque o match da corrente
evita apenas a busca linear residual; o teto do slow path inteiro é **0,60 ms/frame** nos 15 s finais,
mas o walker/resolve continua obrigatório. O antigo intervalo de 0,2–2,5 ms não deve ser atribuído
ao fastpath.

---

## 5. Dynamic state — [ready]

### 5.1 Desperdício confirmado

- 4.527.205 draws;
- 4.527.204 misses de dynamic state e zero hits;
- 2.215.915 commits vazios, 48,94% dos draws;
- custo: **0,258 ms/frame** na sessão, **0,305 ms/frame** nos 15 s finais.

Reasons amostrados e extrapolados:

- generation apenas (`0x1`): aproximadamente 3,42 M, **75,6%**;
- generation + pipeline (`0x3`): aproximadamente 1,05 M, **23,1%**;
- combinações com indexed/feedback: aproximadamente 56,6 mil, **1,25%**.

O grupo pendente dominante é somente depth/stencil (`0x2`), aproximadamente 2,18 M vezes. Desses,
aproximadamente 2,00 M não emitem nenhum comando e 0,179 M emitem algo. Os demais grupos são muito
menores e, quando realmente dirty, tendem a emitir.

Isso prova a cadeia causal:

```text
generation gráfica global muda
  -> todos os cinco helpers são reexecutados
  -> depth/stencil é marcado para comparação em massa
  -> na maioria desses casos Commit não encontra valor Vulkan novo
```

### 5.2 Correção pronta

Usar generations independentes por grupo, com metadata compacta e contígua:

| Grupo | Dependências mínimas lidas hoje |
|---|---|
| viewport/scissor | `screen_scissor`, `window_scissor`, `generic_scissor`, `window_offset`, `viewports[]`, `viewport_scissors[]`, `viewport_control`, `mode_control.vport_scissor_enable`, `clipper_control`, `primitive_type` via `IsClipDisabled` |
| depth/stencil/bias | `depth_control`, formato/base de `depth_buffer` usados por `DepthValid/StencilValid`, `depth_render_control`, bounds, campos de bias em `polygon_control`, `poly_offset`, `stencil_control`, refs front/back |
| primitive | restart enable, `primitive_type`, cull/front em `polygon_control`, `clipper_control.dx_rasterization_kill` |
| rasterization | `line_control` |
| blend/write | `blend_constants`, write masks da key do pipeline, attachment feedback loop |

Guardrails da implementação:

- construir uma tabela `consteval` dword -> grupos e alimentá-la somente com dwords que realmente
  mudaram nos handlers de register write;
- auditar config/context/sh/uconfig, writes indiretos e `ClearState`;
- no slow path, testar os dwords relevantes da interseção; sobreposição de range sozinha gera falsos
  dirty;
- pipeline pointer não invalida todos os grupos: neste bloco somente write masks relevantes mudam
  blend/write;
- `is_indexed`, restart index e `enable_window_offset` preservam seus validators, mas não devem sujar
  grupos Vulkan sem valor emitido dependente;
- novo command buffer continua chamando `DynamicState::Invalidate()` e força reemissão do estado
  cached, mesmo sem generation nova;
- falsos positivos são aceitáveis para a primeira versão; falsos negativos não.

Forma desejada no assembly do hit:

```text
comparar generation de cada grupo
  -> chamar somente helpers cujas generations divergiram
  -> Commit compacto
```

Não é necessário alterar o contrato Gnm nem a semântica do estado emulado. O ganho realista é
**0,15–0,25 ms/frame**, com teto de **0,305 ms/frame** no trecho final.

---

## 6. Descriptors — reutilização cross-pipeline refutada

### 6.1 Fatos do v7

- a amostra cross-pipeline contém 17.154 trocas elegíveis;
- `descriptor_cross_pipeline_exact_state_hits = 0`;
- `descriptor_cross_pipeline_compatible_layout_hits = 6.003` (**35,0%**);
- `descriptor_cross_pipeline_reusable_hits = 0`;
`descriptor_emit` e compare permanecem trabalho mensurável, mas a amostra não encontrou uma única
troca que satisfizesse simultaneamente igualdade de estado e layout compatível. Logo, reusar o
descriptor set apenas pela troca de pipeline não tem base nesta cena. O total de
`descriptor_prepare` tem caudas de preempção e não deve ser convertido em ganho sem outro mecanismo.

### 6.2 Decisão

Os timers v7 separam:

- `descriptor_user_data` — `MakeUserData` e `PushUd`;
- `descriptor_buffers` — preparação/finalização de buffers;
- `descriptor_textures` — resolução/binding de imagens e samplers;
- `descriptor_capture` — cópia do snapshot usado no draw seguinte.

Em uma amostra 1/64 das trocas de pipeline elegíveis no mesmo command buffer/epoch, a v7 executou
uma comparação diagnóstica completa e registrou os counters acima. A assinatura de layout compatível
isolada mede prevalência, mas não autoriza reuso sem igualdade de estado e key exata.

“Compatible layout” no CSV significa igualdade de uma assinatura diagnóstica de 64 bits construída
com bindings, tipos, counts, stage flags e modo push. Ela mede prevalência; não será usada sozinha
como autorização semântica. Uma correção futura deve usar uma key exata ou layouts internados e
respeitar as regras Vulkan de pipeline-layout compatibility.

Não há implementação cross-pipeline pronta: `reusable_hits / samples = 0`, portanto qualquer
fastpath teria custo e risco sem trabalho evitável demonstrado. O potencial de **0,5–1,5 ms/frame**
foi removido da tabela executiva; timers de preparação permanecem úteis apenas para uma hipótese
independente, como reduzir resolução de texturas, que ainda não está formulada.

---

## 7. Readback, EOS/EOP e submits — rebaixado

### 7.1 O que o v7 confirmou

Os 27.284 registros completos fecham exatamente:

- 24.682 (`90,46%`) `event_write_eos` produzidos por `compute_dispatch`;
- 2.602 (`9,54%`) `event_write_eop` produzidos por `graphics_draw`;
- uma imagem em todos os drains úteis;
- zero `same_epoch`;
- 24.682 storage images e 2.602 render targets;
- nenhum caso de reenqueue da mesma imagem/epoch.

Drains:

- guest submit: 2.602 calls e zero candidatos;
- EOS: 26.748 calls, 24.682 candidatos/agendados;
- EOP: 39.118 calls, 2.602 candidatos/agendados.

Portanto, não existe nesta run evidência de reenqueue da mesma imagem/epoch, e não existe conjunto
de várias imagens dentro de um drain para batching local. Essas ideias saem do backlog ativo.

### 7.2 Contrato do SDK e correspondência observada

O SDK oficial estabelece:

- `writeAtEndOfShader(kEosCsDone)` espera apenas o trabalho CS anterior do graphics pipe;
- para `kEosCsDone`, o comando deve vir imediatamente após o dispatch, ou o comportamento é
  indefinido;
- EOS não espera outros estágios nem o async compute pipe;
- EOP é associado ao batch imediatamente anterior e só sinaliza após a condição/evento e cache
  actions pedidos;
- `waitOnAddress` impede o command buffer consumidor de avançar até o valor esperado existir.

O dump corresponde a esse contrato: EOS drena storage escrita por compute; EOP drena render target
escrito por graphics. Pular, fundir ou mover genericamente esses boundaries continua incorreto.

### 7.3 Custo direto sustentado

Submits induzidos pelo readback:

- EOS: 24.682 calls, **68,5225 ms** de mutex wait e **578,6399 ms** no driver;
- EOP: 2.602 calls, **5,4980 ms** de mutex wait e **134,4514 ms** no driver.

Combinados, o driver custa aproximadamente **0,207 ms/frame** e o mutex wait **0,028 ms/frame**.
O dump não mede observador de fence/endereço nem distância até o próximo submit
natural, portanto um deferral semântico ainda não está pronto.

Estimativa revisada: **0,2–0,5 ms/frame** diretamente plausíveis, mais eventual efeito de fila não
isolado. A antiga faixa de 5–12 ms não é sustentada por esta run e foi removida.

---

## 8. Outros mecanismos medidos e rebaixados

### 8.1 Submit/present mutex

- submit mutex wait total: 74,94 ms, aproximadamente 0,029 ms/frame;
- present mutex wait não foi material;
- submit driver total: 729,75 ms, distribuído entre submits normais e readbacks;
- a espera do produtor continua pequena frente ao trabalho de draw/staging.

Estreitar o lock pode ser correto como limpeza futura, porém não explica a queda da cena nem promete
ganho significativo.

### 8.2 Event query e memory notify

- 3.332.382 queries;
- todas encontraram zero watch ativo;
- 26.659.056 pares de counter examinados;
- timer estimado: aproximadamente **0,052 ms/frame** para stores e **0,080 ms/frame** para notify;
- 3.451.107 notifications de memória, 3.380.396 sem watch e 70.711 com watch ativo;
- `memory_notify` foi aproximadamente **0,117 ms/frame** no agregado.

É trabalho real, mas o teto é baixo e a instrumentação detailed aumenta sua participação. Não é um
alvo principal da queda.

### 8.3 WAIT_REG_MEM — já watch-driven, sem promoção

O caminho de `WAIT_REG_MEM` já arma watches por página/endereço quando há uma espera bloqueante e
acorda por notificações relevantes; a busca periódica observada no dump não é evidência de que o
emulador esteja fazendo um polling cego contínuo. Nesta run, as queries de evento tiveram zero watch
ativo, enquanto 70.711 notifications encontraram watch ativo. Portanto, não há base para substituir
o mecanismo por outro scheduler ou por uma tabela de tokens especulativa. O custo restante é parte
do envelope de event-query/memory-notify, com teto baixo e sem ganho significativo demonstrado.

### 8.4 Image lookup/render targets

Amostras de `FindImage` (amostragem 1/32, extrapolada):

- exact cache: aproximadamente **9,64 M**, **91,15%**;
- perfect scan: aproximadamente **924,8 mil**, **8,74%**;
- overlap: aproximadamente **8,7 mil**, **0,08%**;
- create: aproximadamente **2,9 mil**, **0,03%**;
- render-target rebinds: zero.

`ImageFind` custa aproximadamente **0,50 ms/frame** no agregado, mas mistura call sites e contém raras
caudas de preempção. Um cache estrutural por attachment só pode pular o que UID, topology epoch,
descrição e backing provarem redundante; não pode pular update, alias handling, layout transition,
writer marking ou lifetime/LRU. Potencial atual: menos de aproximadamente 0,3–0,5 ms/frame, sem
prioridade sobre staging e dynamic state.

### 8.5 Decode PM4/register writes

O volume PM4 é alto, mas nenhum timer atual isola decode/handler de draw e staging. O envelope
`gcp_active - draw - dispatch` inclui decode, submits, readback e instrumentação simultaneamente;
não é ganho atribuível a PM4. Os fast paths de write já comparam dwords realmente alterados. Uma
mudança adicional só será promovida quando houver custo por opcode/handler, não apenas redundância.

---

## 9. Schema v8 e como avaliar a próxima run

O dump auditado nesta revisão é **schema v7**. A build corrente que deve gerar a próxima run é a
**v8** (`shadps4-perf.exe`); o schema foi incrementado de 7 para 8. Não há I/O no hot path; todos
os dados permanecem em memória e o CSV é escrito apenas no shutdown normal via `Alt+F4`.

### 9.1 Novas perguntas respondidas

| Área | Nova evidência | Decisão que ela permite |
|---|---|---|
| staging | dedup zero no v7; v8 separa preparação, sparse copy e plano mapped/zero/fallback | escolher a transformação de maior custo removível |
| SRT | 97,18% dos slow results são `current_hits` | implementar apenas o match pós-resolve exato; walker continua obrigatório |
| descriptors | 0 exact-state e 0 reusable hits; 35,0% só de layout compatível | retirar reutilização cross-pipeline do backlog |
| WAIT_REG_MEM | watches ativos acordam por notifications; queries sem watch não dominam | manter watch-driven; não inventar scheduler novo |

### 9.2 Regras de leitura

- descartar o primeiro frame;
- usar counters globais para fechamento da sessão;
- usar os 15 s finais para a cena indicada;
- multiplicar timers pelos períodos declarados em `timer_metadata`;
- multiplicar counters de batch staging e cross-pipeline pelos períodos em `metadata` quando for
  necessária uma estimativa de calls; razões como requests/sample podem usar diretamente as somas;
- os counters `staging_sparse_*` usam o período 16 e o mesmo gate dos batches contendo Guest de
  `staging_batch_samples`; batches apenas Host/Zero não emitem um registro sparse. O timer
  `staging_sparse_copy` usa seu período próprio 64;
  `staging_sparse_copied_bytes = staging_sparse_mapped_bytes + staging_sparse_zero_bytes`, incluindo
  os fallbacks `copy_reference`; `staging_sparse_reference_bytes` é subconjunto e não deve ser
  somado novamente;
- não somar fases ao timer pai;
- comparar histogramas e médias por chamada para detectar preempção;
- verificar se os novos counters fecham com `stage_permutation_hits`, descriptor pipeline reasons e
  staging allocations dentro do erro determinístico de amostragem.

### 9.3 Critérios para promover hipóteses

**Staging [ready].** O v7 refutou dedup, e o v9 isolou resolução + payload, fechou os invariantes e
definiu a transformação host-first descrita na seção 13.

**SRT [ready].** A distribuição current/other já determinou o micro-fastpath pós-resolve; manter
walker/resolve e igualdade exata, sem pular o walker SRT.

**Descriptors** está refutado para reuso cross-pipeline nesta cena porque a interseção foi zero.

---

## 10. Ordem de implementação futura

1. **Staging host-first — [ready].** Maior ganho provável; implementar nas fases isoladas da seção
   13, preservando snapshot, sparse fallback e lifetime.
2. **Dynamic generations por grupo — [ready].** Menor ganho absoluto, mas mecanismo totalmente
   delimitado e risco controlável.
3. **SRT pós-resolve — [ready], micro-fastpath.** Comparar a especialização materializada com a
   `current_permutation`; nunca pular walker/resolve nem cobrir TCS/TES sem key completa.
4. **Descriptors cross-pipeline — refutado nesta cena.** Não reabrir sem uma mudança de workload.
5. **Readback deferral.** Reabrir apenas com endereço/observador e distância ao próximo submit; não
   usar epoch/batching já contrariados por esta run.

Após cada otimização:

- conferir o assembly Release AVX2 gerado, sem inline assembly;
- medir o counter de trabalho removido e o timer interno na detailed;
- testar FPS/framedrops na Release, mesma cena;
- validar outros jogos e integridade visual;
- não combinar duas mudanças antes de atribuir o ganho da primeira.

---

## 11. Checklist data-oriented

| Dimensão | Aplicação neste caso |
|---|---|
| L1/L2/L3 | generations e snapshots compactos por grupo/programa; evitar reconstruções, vectors e pointer chasing por draw |
| SIMD | AVX2 somente em runs contíguos e comparações exatas comprovadas; nunca overread de memória guest |
| batching | stream batch já existe; medir canonicalização antes de ampliá-lo; readback não tem múltiplas imagens por drain nesta run |
| fluxo coerente | manter o caminho comum linear; validators e failures continuam frios/`SHAD_NO_INLINE` |
| hot/cold | telemetria amostrada, cardinalidade fixa e sem I/O; build Release elimina os gates detailed |
| NUMA | não há evidência de problema NUMA; o gargalo é trabalho serial do GCP, não alocação remota medida |

---

## 12. Conclusão provisória do v7 — substituída pela seção 13

O dump v7 confirma que a queda ao mover é amplificação de workload: mais draws, PM4 e staging por
frame, com custo unitário estável. O maior scope recorrente observado é `ExecuteStreamCopyBatch`,
chegando a 5,22 ms/frame no trecho final. Naquele ponto, sua fração evitável ainda exigia a
decomposição posterior, concluída pelo schema v9 na seção 13.

Dynamic state e SRT são as hipóteses já **[ready]**: a generation global força recomputação em todo
draw, e depth/stencil domina trabalho que frequentemente termina sem emissão; no SRT, 97,18% dos
slow results permitem o micro-fastpath pós-resolve. O ganho SRT continua estimado em 0,01–0,02
ms/frame após o walker/resolve; o skip do walker continua inseguro. A hipótese
de reuso cross-pipeline de descriptors foi refutada pelos zero reusable hits. Readback foi corretamente
rebaixado: os 27.284 registros completos mostram epochs sempre novos, uma imagem por drain e
boundaries compute/graphics coerentes com o SDK.

Os dumps v8/v9 cumpriram esse gate: staging agora é implementável sem alterar ordering, estado
emulado ou comportamento do jogo, sob os invariantes da seção 13.7.

---

## 13. Schema v9 — staging fechado e arquitetura recomendada [ready]

Esta seção substitui, para staging, o estado provisório `[needs-v8]` das seções 0, 3, 9, 10 e 12.
O schema v9 decompõe o custo até o payload e fornece informação suficiente para implementar a
correção sem presumir que deduplicação, coalescing, fence ou memória esparsa dominam.

### 13.1 Proveniência e integridade do dump decisivo

- arquivo: `C:/Users/Arthur/AppData/Roaming/shadPS4/log/shadps4-telemetry-1786812366836.csv`;
- schema: v9;
- tamanho: 52.320.336 bytes;
- SHA-256: `40FB1C9B7F90C2AD46E5F000AFE23E96B9EC6500A6CF3D45A353D857712E4EA9`;
- 814.332 linhas incluindo o header, sem linha malformada;
- duração: 51,9190492 s;
- 2.497 frames, sem overwrite de frame;
- 25.777 registros de writeback, sem overwrite;
- os 11.816 overwrites pertencem somente ao ring genérico de eventos e não afetam counters,
  frames, timers agregados nem amostras sparse;
- os seis timers do gate compartilhado sparse possuem exatamente 65.250 amostras cada e não há
  divergência de sample count por frame;
- todos os invariantes sparse fecham exatamente.

Os dois dumps v8 imediatamente anteriores reproduzem o mesmo envelope no trecho final: 4,90 e
5,19 ms/frame em `staging_sparse_copy`, contra 4,72 ms/frame no v9. O resultado v9 não é uma cauda
isolada de uma única run.

### 13.2 O mecanismo causal está fechado

Nos 15 s finais, identificados pelo usuário como a janela relevante:

| Métrica | Todos (740) | Rápidos `<=18 ms` (578) | Lentos `>=30 ms` (160) |
|---|---:|---:|---:|
| frame | 20,270 ms | 16,467 ms | 34,039 ms |
| draws/frame | 1.947 | 1.645 | 3.035 |
| batches sparse estimados/frame | 1.934 | 1.632 | 3.018 |
| requests guest estimados/frame | 8.145 | 6.964 | 12.391 |
| bytes guest sparse/frame | 16,855 MB | 14,297 MB | 26,073 MB |
| staging total/frame | 23,243 MB | 18,428 MB | 40,698 MB |
| `staging_stream_batch` | 4,885 ms | 4,153 ms | 7,519 ms |
| `staging_sparse_copy` | 4,716 ms | 3,991 ms | 7,341 ms |
| tempo médio por sample sparse | ~2,44 us | ~2,44 us | ~2,43 us |

O custo por batch não degrada no frame lento; o número de batches, requests e bytes aumenta. Nos
15 s finais, a correlação de frame time com staging total é 0,817, com draws é 0,807 e com o timer
sparse é 0,745. A correlação entre bytes guest e o timer sparse é 0,917. Uma regressão simples do
timer pai contra bytes resulta em aproximadamente 3,62 GB/s efetivos e intercepto de apenas
0,060 ms/frame. Isso confirma um custo unitário estável multiplicado por workload, não uma espera
episódica ou uma mudança de modo do driver.

O batch médio contém 4,21 requests guest de 2.069 bytes. A distribuição global de alocação é
concentrada entre 2 e 32 KiB; batches `>=8 KiB` são 33,79% das chamadas e representam
aproximadamente 75,3% dos bytes usando o centro de cada bucket; batches `>=16 KiB` são 18,33% das
chamadas e aproximadamente 56,3% dos bytes. Essa assimetria é importante para uma política híbrida:
não se deve pagar um comando Vulkan extra para todo batch pequeno.

Fatos que eliminam hipóteses anteriores:

- requests e canonical copies são idênticos: zero exact reuse e zero subrange reuse dentro do
  draw;
- nos 15 s finais há zero pares e zero bytes mergeable; na sessão inteira são somente 135 pares em
  1.100.357 requests, ou 67,5 KiB em 2,30 GB;
- 100% dos bytes guest são mapped, com exatamente um mapped run por request e zero zero-fill;
- não há `copy_reference` fallback;
- 56,57% dos lookups de plano missam; nos 15 s finais todos os misses são conflito, não mudança de
  generation, entrada vazia nem request não cacheável;
- somente 3,36% dos runs usam non-temporal stores, embora cubram 25,42% dos bytes; finish/sfence é
  pequeno;
- `stream_slice_hits`, `stream_slice_misses`, `buffer_token_hits` e `buffer_token_misses` são zero:
  o workload atual passa pelo batch novo e não pelo mecanismo antigo de reuse de slices.

Portanto, o trabalho removível está em três lugares, nesta ordem conceitual: resolver como esparso
um intervalo que quase sempre é denso; escrever sincronicamente milhares de pequenos blocos no
backing escolhido para `MemoryUsage::Stream`; e recopiar, entre draws do mesmo tick, fontes que
podem se repetir. Deduplicação local, merge de runs e fence não explicam o custo.

### 13.3 Decomposição do v9

Os timers do gate compartilhado usam um sample determinístico diferente do timer pai
`staging_sparse_copy`. Por isso suas proporções internas são válidas, mas seus valores absolutos não
devem ser subtraídos diretamente do pai. Normalizando a proporção do gate pelo timer pai, o frame
lento fica aproximadamente assim:

| Fase | Medida no gate | Equivalente normalizado ao pai | Participação no gate |
|---|---:|---:|---:|
| aquisição do shared lock | 0,086 ms | 0,124 ms | 1,69% |
| lookup do plano | 0,414 ms | 0,600 ms | 8,17% |
| construção do plano | 0,424 ms | 0,614 ms | 8,37% |
| cópia do payload | 3,051 ms | 4,415 ms | 60,15% |
| finish/sfence | 0,061 ms | 0,088 ms | 1,20% |
| residual do scope | 1,036 ms | 1,499 ms | 20,42% |

O residual interno do gate inclui loop/control flow e timestamps da própria detailed; ele não é
inteiramente eliminável. A liberação do shared lock ocorre **depois** da captura de
`sparse_shared_ns` e antes do fim do timer pai. Logo, ela pertence ao gap pai--gate, não à linha de
residual da tabela. Como os gates usam samples distintos, esse gap não pode ser isolado
quantitativamente neste dump. Mesmo assim, lookup + build têm um envelope normalizado de
aproximadamente 1,21 ms nos frames lentos, enquanto payload tem 4,42 ms. O desenho precisa atacar
ambos; otimizar apenas a tabela de planos deixa a maior parte do ganho na mesa.

Como comparação interna útil, o caminho `staging_image` já escreve em `MemoryUsage::Upload`. Ele
move 5,25 MB/frame em 0,476 ms no trecho final e 13,09 MB/frame em 0,994 ms nos frames lentos,
aproximadamente 11--13 GB/s efetivos. Seus requests são muito maiores, então isto não é um A/B
controlado de tipo de memória; ainda assim, demonstra que um backing host-preferred existente é
capaz de receber o mesmo tipo de memória guest muito mais rapidamente que o envelope de pequenos
stream copies.

### 13.4 O que o código e o assembly realmente fazem

`Rasterizer::BindResources` abre um batch antes de preparar buffers de todos os estágios.
`PrepareVertexIndexBuffers` acrescenta vertex/index e `FinalizeStreamCopyBatch` captura os bytes
antes de `scheduler.BeginRendering` e antes do draw/dispatch consumidor. O batch já é, portanto, o
maior batch semanticamente gratuito: todos os recursos transitórios de um draw. Ampliá-lo para
vários draws sem antes capturar os bytes mudaria o instante observado da memória guest.

Somente buffers read-only, de até 64 KiB e sem `GpuModified` entram no stream guest. A cópia é uma
materialização de snapshot para um buffer Vulkan; o endereço Gnm original não é exposto diretamente
ao shader host.

O caminho Release sem telemetria confirma o excesso de frontend:

- `MemoryManager::CopySparseMemoryBatch`: aproximadamente 2.756 bytes, 654 instruções,
  127 branches condicionais, 18 calls, oito registradores nonvolatile salvos e 0x98 bytes de stack;
- `BufferCache::ExecuteStreamCopyBatch`: aproximadamente 2.317 bytes, 557 instruções,
  106 branches condicionais, 14 calls e oito registradores nonvolatile salvos.

No primeiro, o LLVM mistura na mesma função o protocolo de `SharedFirstMutex`, hash/tag do plano,
walker completo da `std::map` de VMAs, montagem/fallback de até quatro runs, kernels cached/NT,
memcpy/memset e sfence. O caso dominante do dump -- um único run mapped -- ainda atravessa uma key
exata por `(source,size)` e, nos misses, o RB-tree. Para os requests comuns próximos de 2 KiB, o
assembly chama `memcpy`; AVX2 inline aparece no tiny path e o loop NT é corretamente vetorizado e
desenrolado, mas isso não corrige o destino nem o trabalho de resolução.

`ExecuteStreamCopyBatch` é grande, porém deduplicate + layout + request prepare + results somam
somente cerca de 0,54 ms nos frames lentos, e parte disso é instrumentação. Uma reescrita ampla da
canonicalização teria risco maior que seu teto. A reforma deve ficar confinada ao backend de
captura e à resolução dense/sparse.

`MemoryUsage::Stream` pede `AUTO_PREFER_DEVICE`, mapping sequencial e `HOST_COHERENT`.
`MemoryUsage::Upload` pede `AUTO_PREFER_HOST`. Na RTX 3060 observada existem:

- 11,83 GiB de heap device-local;
- 11,96 GiB de heap host;
- somente 214 MiB de heap simultaneamente device-local + host-visible + coherent;
- um memory type device-local/host-visible/coherent nesse heap de 214 MiB;
- memory types host-visible/coherent no heap host, incluindo uma variante host-cached.

É uma **inferência suportada**, ainda não um fato logado por VMA, que o ring `Stream` de 64 MiB seja
alocado no heap BAR de 214 MiB: ele é o único tipo que satisfaz simultaneamente preferência por
device-local, mapping e coherence. Nesse caso o GCP escreve VRAM através de PCIe e fica bloqueado
nessas stores. A implementação deve registrar uma vez o heap index e os property flags reais de
cada ring para converter essa inferência em fato antes de escolher a política.

### 13.5 Contrato do SDK oficial

Os headers oficiais `sdk/target/include_common/gnm/buffer.h` e `gnm/constants.h` estabelecem que o
buffer descriptor contém base address, tamanho/stride e `ResourceMemoryType`; o memory type controla
cache e bus, não acrescenta uma operação implícita de snapshot. `SC` é system-coherent e recomenda
Onion; `PV`, `GC` e `RO` recomendam Garlic. O manual oficial separa explicitamente:

- memória CPU-only: WB Onion;
- memória GPU-only: WC Garlic;
- memória acessada por CPU e GPU: WB Onion fortemente recomendada em um caso oficial;
- buffers read-only usados simultaneamente por CPU/GPU podem usar WB Garlic, mas somente depois de
  mudar proteção/tipo e sincronizar o fim do uso anterior pela GPU.

O tutorial oficial de WB Garlic usa double buffering e só muda o papel do buffer após EOS/EOP ou
evento equivalente. Isso corrobora dois invariantes do emulador: backing pode ser escolhido conforme
quem o acessa, mas o conteúdo e o lifetime não podem ser reutilizados depois de o produtor ou o
consumidor avançar sem sincronização.

O SDK não prova qual memory type o VMA escolherá numa GPU de PC e não mede PCIe. Ele sustenta a
semântica da solução: um ring host para bytes recém-produzidos pela CPU e um ring device-local para
bytes promovidos à GPU são representações válidas do mesmo buffer read-only, desde que snapshot,
visibilidade e lifetime sejam preservados.

### 13.6 Arquitetura escolhida: transient read stream host-first

A melhor relação ganho/risco não exige um novo subsistema global. O desenho pode permanecer privado
a `BufferCache`, reutilizando `StreamBuffer`, `StreamCopyRequest`, `StreamCopyResult` e o scheduler.
Conceitualmente há um `transient read stream` com duas camadas obrigatórias e uma terceira
contingente:

1. **captura CPU em ring host-preferred dedicado**;
2. **consumo direto desse ring pela GPU**;
3. **promoção opcional para um ring device-local dedicado, somente quando puder ser gravada sem
   criar uma transição de rendering adicional**.

O fluxo por draw deve ser:

```text
BeginStreamCopyBatch
  -> coletar shader + vertex/index requests como hoje
FinalizeStreamCopyBatch  [o snapshot continua exatamente aqui]
  -> resolver reuse seguro do mesmo tick, quando existir
  -> compactar somente os misses preservando ordem/alinhamento
  -> capturar misses guest no ring host-preferred
       -> fast path de intervalo mapped
       -> fallback sparse frio para PRT/unmapped
  -> backend:
       pequeno/normal: bind direto do ring host
       grande, vantajoso e fora de rendering: uma promoção contígua host -> device-local + barrier
  -> materializar um StreamCopyResult por request
  -> draw/dispatch consumidor
```

#### Camada A — fast path de intervalo mapeado

Substituir o cache direto de planos exatos no caminho comum por um micro-cache thread-local de
intervalos mapped, invalidado por `owner + mapping_generation`. Uma entrada precisa somente de
`begin/end`; o batch lê a generation uma vez sob o shared lock. Se
`source >= begin && size <= end - source`, o request é uma cópia densa direta.

No miss, uma função `SHAD_NO_INLINE` fria usa `FindVMA`, estende o intervalo através de VMAs
contíguas mapped e preenche o micro-cache. Somente um request que cruza mapped/unmapped continua no
planner/walker sparse atual. Um last-hit seguido de quatro ou oito intervalos compactos é preferível
a ampliar a tabela atual de planos de 48 bytes: os intervalos cobrem muitas combinações
`(source,size)` sem pressionar L1.

O hot assembly desejado por request é: range check sem overflow, chamada/inline do kernel de cópia
adequado e avanço para o request seguinte. RB-tree, construção de run array, validação rara e
fallback ficam fora da função quente. O shared lock por batch permanece; seu custo medido é pequeno
e removê-lo abriria race com map/unmap.

#### Camada B — ring host-preferred dedicado

Adicionar um `StreamBuffer` dedicado com `MemoryUsage::Upload` e `AllFlags` somente para snapshots
read-only de draw. Não reutilizar o `staging_buffer` geral: separar os rings evita interferência de
lifetime, estado de barrier e pressão de wrap com image/upload traffic. Não alterar o
`stream_buffer` atual nem a política global de `MemoryUsage::Stream`, usada também por outros
utility buffers como tile/GDS.

O caminho padrão em GPU discreta deve copiar guest diretamente para esse ring e devolver seu
`Buffer*`/offset em `StreamCopyResult`. Isso troca stores síncronas CPU->BAR por leituras da GPU a
partir de host memory. A taxa alvo é modesta: 26,1 MB por frame lento equivale a 1,57 GB/s a 60 fps,
bem abaixo da largura de banda prática de PCIe do host.

O dump registra `gpu_idle_gap_ns` médio de 13,99 ms/frame nos 15 s finais, 81,67% do wall time nos
frames rápidos e 46,87% mesmo nos frames lentos. Esse counter mede o intervalo entre completion
observado e o próximo submit, não um timestamp GPU exato; a medição externa do usuário aponta mais
de 75% de ociosidade. Em conjunto, há evidência suficiente para transferir trabalho ao consumidor
GPU, mas a decisão final continua condicionada a GPU timestamps e FPS Release.

Em UMA ou quando VMA já escolher memória host-cached/device-local rápida, a política pode manter o
ring atual. A escolha deve usar property flags/heap reais, não vendor string.

#### Camada C — promoção GPU opcional, não por request nem à custa de renderpass break

Se bind direto de host memory aumentar materialmente o tempo GPU, batches grandes podem ser
promovidos para um ring device-local **dedicado**:

- reservar o mesmo layout contíguo nos dois rings;
- copiar guest apenas para o ring host;
- emitir um único `vkCmdCopyBuffer` para todo o bloco, incluindo padding não observado;
- emitir depois uma única dependency `TRANSFER_WRITE -> MEMORY_READ/ALL_COMMANDS` para o range;
- devolver offsets do ring device-local.

O `stream_buffer` atual não serve como destino garantidamente device-local: `MemoryUsage::Stream`
continua mapped e apenas *prefere* device memory. O `device_buffer` geral também não deve ser
reutilizado como ring transitório, pois hoje é compartilhado por outros caminhos e não oferece uma
reserva non-mapped com watches de lifetime. A extensão local mínima é separar em `StreamBuffer` a
reserva/lifetime (`Reserve`) do acesso CPU (`Map`), e instanciar um ring `MemoryUsage::DeviceLocal`
privado deste backend. `Map` continua sendo `Reserve + mapped pointer`. O commit também precisa ser
separado em `CommitHostWrites`, que faz flush quando necessário e conclui a reserva, e
`CommitReservation`, que apenas avança os watches; chamar o flush atual sobre memória non-mapped
device-local seria incorreto. O ring de promoção usa `Reserve + CommitReservation`.

Há ainda uma restrição mais forte que tamanho: `FinalizeStreamCopyBatch` pode executar enquanto o
scheduler mantém aberto o rendering scope do draw anterior. `vkCmdCopyBuffer` não pode ser gravado
dentro dele. Chamar `scheduler.EndRendering()` em cada promoção introduziria milhares de
renderpass/dynamic-rendering breaks e provavelmente anularia o ganho. Portanto, promoção só é
admissível quando o rendering já será encerrado pela transição seguinte, ou depois de uma medição
mostrar que o break adicional custa menos que o acesso host direto. Na ausência desse caso, o
backend correto permanece host-direct mesmo para batches grandes.

Não emitir copy + barrier para todo batch de 2--4 KiB. O custo de gravação de dois comandos Vulkan
milhares de vezes por frame pode devolver o gargalo à CPU. O threshold inicial deve ficar em 8 ou
16 KiB e ser validado **depois** do gate de rendering; `>=16 KiB` cobre aproximadamente 56% dos
bytes com apenas 18% dos batches.
Requests cujo offset/tamanho não satisfizer os múltiplos de quatro exigidos por `VkBufferCopy`
ficam no backend direto ou usam tamanho de alocação arredondado, nunca overread fora da reserva.

Essa promoção não altera o instante do snapshot: os bytes já foram capturados no ring host antes de
qualquer adiamento. Ela apenas desloca a segunda transferência para a GPU ociosa.

#### Camada D — reuse cross-draw limitado ao mesmo tick

O batch atual prova somente que não há duplicata **dentro** de um draw. O cache de planos encontra a
mesma key `(source,size)` em 43,43% dos requests apesar de ter apenas 64 slots direct-mapped; com
milhares de requests por frame, isso é forte evidência de recorrência local, mas não prova igualdade
do conteúdo.

O `StreamSliceReuseState` existente deve ser integrado ao batch como camada opcional e adaptativa:

- key exata por source type/address/size;
- offset precisa satisfazer o alinhamento novo;
- mesma `StreamBuffer::Generation()` e mesmo `scheduler.CurrentTick()`;
- igualdade byte a byte antes de autorizar o hit; hash sozinho nunca autoriza reuse;
- após mismatch repetido, cooldown para não fazer `memcmp`/shadow em buffers voláteis;
- hits são resolvidos antes do layout, e apenas misses ocupam o novo bloco.

O limite ao mesmo tick é obrigatório. Os watches atuais do ring protegem o lifetime da alocação no
tick em que `Commit` a criou; reutilizar o offset em tick posterior não estende esse lifetime. Reuse
cross-tick só poderá existir depois de um `TouchRange`/lease cujo algoritmo preserve a ordenação
monotônica dos upper bounds, ou por meio de um buffer persistente separado.

Uma evolução por PageManager write-watch poderia eliminar o `memcmp` de entradas comprovadamente
estáveis, mas não pertence à primeira correção: armar watches por página pode introduzir page faults,
fan-out e contenção nos escritores. O mecanismo exato já existente tem menor mudança arquitetural e
falha para o caminho normal quando o conteúdo muda. A arquitetura proposta deixa o ponto de reuse
antes do layout, portanto write-watch pode ser acrescentado depois sem reescrever o backend.

### 13.7 Invariantes obrigatórios da implementação

1. **Snapshot:** os bytes de cada miss são capturados em `FinalizeStreamCopyBatch`, antes do draw ou
   dispatch correspondente. Não acumular fontes guest cruas através de draws.
2. **Autoridade:** somente requests já classificados read-only e `!GpuModified` usam esse caminho;
   GPU-written buffers continuam no buffer cache normal.
3. **Sparse/PRT:** range cache autoriza somente intervalo inteiramente mapped na mesma
   `mapping_generation`; qualquer dúvida cai no walker atual, preservando zero-fill e ausência de
   overread.
4. **Map/unmap:** o shared lock cobre resolução e cópia; a generation é observada sob esse lock.
5. **CPU visibility:** non-coherent upload é flushed em `Commit`; NT stores recebem um único sfence
   antes do commit/uso. Não adiar ou remover esse fence por batch.
6. **GPU visibility:** bind direto preserva o domínio host->device do submit. Promoção registra copy
   antes do consumidor e barrier de transfer-write para todos os tipos de leitura possíveis.
7. **Lifetime:** ring host precisa viver até a leitura direta ou a transfer copy completar; ring
   device-local dedicado precisa viver até o draw completar. Ambos usam os ticks/wrap waits atuais;
   a reserva non-mapped não pode contornar os watches.
8. **Alinhamento:** cada resultado conserva o alinhamento Vulkan pedido; offsets usados por
   descriptors e index/vertex bindings continuam válidos. Aritmética de range não pode overflowar.
9. **Reuse:** conteúdo exato, generation do ring, tick e alinhamento fazem parte da autorização.
   Um hit nunca é inferido apenas por endereço, hash, plan hit ou ausência presumida de escrita.
10. **Resultado misto:** `StreamCopyResult` já contém `Buffer* + offset`; um draw pode receber hits de
    uma slice anterior e misses do ring atual sem mudar a API do rasterizer.

### 13.8 Alternativas rejeitadas ou rebaixadas

| Alternativa | Decisão |
|---|---|
| aumentar o cache exato de planos | rejeitado: consome L1 e continua usando uma key estreita; o cache de intervalo cobre a causa |
| dedup/subrange dentro do draw | refutado por zero hits |
| coalescer requests adjacentes | refutado por zero pares nos 15 s finais |
| remover sparse handling globalmente | incorreto para PRT e outros jogos; manter fallback frio |
| adiar a captura guest até submit | incorreto: altera o snapshot entre draws |
| remover sfence/flush | ganho medido mínimo e risco direto de corrupção |
| trocar `SharedFirstMutex` globalmente | sem base: acquire é ~0,12 ms normalizado no frame lento |
| custom memcpy/AVX2 como correção principal | rebaixado: o assembly já usa AVX2/NT; backing e bytes evitados dominam |
| importar toda memória guest via `VK_EXT_external_memory_host` | grande reforma, extensão não universal e alto risco de alias/lifetime/coherency |
| usar sempre GPU copy para todo batch | rebaixado: milhares de comandos/barriers e rendering breaks por frame podem consumir o ganho CPU |
| usar sempre device-local ou sempre host em todo hardware | rejeitado: a política deve considerar heap real e custo GPU |

Após a troca de backing, vale reavaliar o kernel de cópia. A condição NT atual exige que o tamanho
inteiro seja múltiplo de 64 apesar de já existir tail handling; relaxá-la pode ampliar cobertura.
Esse é um ajuste posterior, pois seu teto é parte dos 4,42 ms de payload e a política ideal muda
quando o destino deixa de ser BAR.

### 13.9 Ganho estimado e ordem de implementação

| Fase | Mudança | Ganho CPU plausível nos frames lentos | Risco |
|---|---|---:|---|
| 1 | micro-cache dense + slow path sparse `SHAD_NO_INLINE` | **0,8--1,4 ms** | baixo |
| 2 | ring host-preferred dedicado, bind direto | **1,3--2,6 ms** | médio; medir tempo GPU |
| 3 | reuse exato adaptativo no mesmo tick | **0--2,5 ms** adicionais, proporcional aos bytes realmente iguais | baixo-médio |
| 4 | promoção GPU somente para batches grandes e sem break adicional, se necessária | preserva o ganho CPU reduzindo eventual penalidade GPU; não é ganho aditivo garantido | médio-alto |

Sem contar reuse, a expectativa combinada responsável é **2,1--4,0 ms** retirados dos frames
lentos. Com conteúdo recorrente estável, **3--5+ ms** é plausível, sempre limitado pelo scope sparse
de 7,34 ms observado. O efeito visual pode ser maior que a redução linear: tirar alguns
milissegundos do produtor pode impedir que o frame atravesse o vblank e apareça como 34 ms.

A ordem recomendada é manter commits isolados e mensuráveis:

1. implementar e validar o resolver dense/cold, incluindo o assembly Release;
2. adicionar o ring host e comparar direct-host contra o ring atual na mesma cena;
3. integrar reuse exato do mesmo tick e medir hit bytes, não apenas hit count;
4. somente se direct-host consumir GPU de forma material, avaliar primeiro a frequência de
   rendering scopes já encerrados; adicionar promoção por threshold apenas nesses pontos, salvo A/B
   explícito que justifique breaks adicionais.

### 13.10 Telemetria e gates da implementação

A implementação deve conservar telemetria somente na detailed, sem I/O no hot path:

- memory type index, heap index e property flags dos rings, registrados uma vez;
- dense-span hits/misses, interval refills, requests sparse reais e bytes de fallback;
- bytes/calls por backend: current stream, host-direct e GPU-promoted;
- número de comandos de promoção, bytes por comando e buckets de tamanho;
- reuse candidates/hits/mismatches/cooldown e, principalmente, bytes evitados;
- timers separados de capture CPU, command recording de promoção e sparse cold path;
- GPU timestamp/busy time ou uma medição equivalente; `gpu_idle_gap_ns` sozinho não é utilização
  precisa;
- `staging_sparse_copy`, `staging_stream_batch`, `draw_cpu_ns`, frame pacing e FPS Release.

Critérios de aceite:

- invariantes sparse continuam fechando e PRT/zero fallback permanece alcançável;
- nenhum readback linear, descriptor, vertex/index ou shader buffer apresenta corrupção;
- queda aparece no timer pai e em `draw_cpu_ns`, não só em counters internos;
- o assembly comum não contém o walker RB-tree nem a montagem de plano;
- a GPU continua abaixo do budget e o número de copy/barrier commands não cria novo gargalo;
- outros jogos continuam corretos nos dois tipos de backing.

### 13.11 Decisão final

**Staging está [ready].** A correção recomendada não é batching adicional nem fence deferral. É uma
reforma local do transient read stream: resolver regiões densas como densas, capturar os misses num
ring host-preferred e permitir consumo direto pela GPU ociosa. Um ring device-local dedicado é uma
promoção seletiva posterior, condicionada a GPU timestamps e à ausência de rendering break
adicional. Reuse exato do mesmo tick entra antes do layout e remove cópias quando os bytes realmente
se repetirem, sem inferir estabilidade.

Esse desenho mantém o boundary e a API do rasterizer, mantém sparse/PRT no caminho frio e limita a
mudança principal a `core/memory.*`, `buffer_cache.*` e extensões locais de property/reserva/lifetime
em `buffer.*`. É a menor mudança arquitetural que ataca simultaneamente o envelope de 1,21 ms de
resolução e o envelope de 4,42 ms de payload sem deslocar cegamente o custo para outra parte do
pipeline.
