# Plano técnico para reduzir o frametime CPU do pipeline PM4 -> Vulkan

## 1. Base, objetivo e critérios de leitura

Este plano foi produzido sobre a branch `codex/performance-opt`, commit `39eae5f4`, e cobre o
pipeline desde a entrada dos command buffers GNM até a emissão e conclusão de draws, dispatches e
apresentação Vulkan. A referência local `upstream/main` estava em `bf4809fe`: a branch estava um
commit à frente e onze commits atrás dessa referência no momento da análise. Portanto, todos os
anchors abaixo são exatos para `39eae5f4`; antes de implementar, é obrigatório atualizar a
referência upstream, comparar os diffs e revalidar anchors e conflitos sem descartar as mudanças
locais já existentes.

Convenções usadas no documento:

- **FATO**: comportamento confirmado diretamente no código ou na documentação oficial do SDK.
- **HIPÓTESE**: oportunidade plausível que ainda precisa de telemetria causal.
- **DECISÃO**: arquitetura escolhida para preservar semântica e permitir implementação incremental.
- **Gate**: condição objetiva para aceitar uma fase e seguir para a dependente.

### Objetivos

1. Diminuir p50, p95 e p99 do frametime CPU e melhorar o 1% low removendo trabalho repetido,
   esperas e chamadas de driver do caminho crítico GCP.
2. Impedir que a GPU fique ociosa esperando o GCP produzir command buffers, dados ou submissões.
3. Preservar exatamente a ordem observável de PM4: writebacks, labels, fences, IRQs, waits,
   aliases, flips e limites de cache.
4. Manter `Rasterizer`, `BufferCache`, `TextureCache`, registradores e decoder com um único owner
   lógico, evitando introduzir locks no hot path.
5. Tornar cada otimização reversível e mensurável isoladamente.

### Não objetivos

- Otimizar execução interna da GPU quando isso não reduz espera CPU/GPU ou custo de gravação.
- Implementar async compute host paralelo sobre caches e `Rasterizer` compartilhados.
- Remover barriers por heurística, tratar todo yield como submit ou trocar correção por throughput.
- Forçar NUMA, afinidade rígida, SIMD manual ou multidraw sem evidência de perfil.
- Unificar imediatamente as timelines dos `Scheduler`; os tokens serão domain-agnostic para que
  essa simplificação possa ser avaliada depois.

Não há previsão percentual de ganho neste plano. Cada ganho alegado deve superar o noise floor do
benchmark e vir acompanhado da redução do contador causal correspondente.

Durante desenvolvimento C/C++, carregar `D:\CODING\SDKs\EWDK\EWDK-LLVM.env` e usar o wrapper
incremental `build-fast.ps1` do projeto (criá-lo se ainda não existir, sem versioná-lo como artefato
pessoal). `build-full.ps1`, clang-format, REUSE e a suíte completa ficam para preparação de PR.
Testes novos são desenhados nessa etapa ou quando solicitados explicitamente; gates intermediários
usam build incremental, traces determinísticos e validação manual da matriz abaixo.

## 2. Fluxo atual PM4 -> draw

### 2.1 Entrada HLE e publicação

1. As APIs GNM constroem pacotes em `src/core/libraries/gnmdriver/gnmdriver.cpp:374-651`.
2. `sceGnmSubmitCommandBuffers` percorre os pares DCB/CCB e chama `Liverpool::SubmitGfx`
   separadamente, incluindo sequências de inicialização também separadas
   (`gnmdriver.cpp:2200-2304`, especialmente `:2236-2301`).
3. Cada `SubmitGfx` pode copiar DCB/CCB, cria uma coroutine `ProcessGraphics`, toma o mutex da fila,
   toma o mutex global de submit e gera um wake (`liverpool.cpp:1336-1385`).
4. Ding-dong compute publica trabalho de ring em `gnmdriver.cpp:315-361`; o ring é power-of-two
   (`:1215-1217`).

### 2.2 GCP, CE/DE e ASC

1. `Liverpool::Process` acorda, varre as filas mapeadas e chama `task.resume()`
   (`src/video_core/amdgpu/liverpool.cpp:204-262`). Há scan e mutex por fila em `:225-236`.
2. Graphics usa dois fluxos cooperativos: CE em `ProcessCeUpdate` (`:264-328`) e DE em
   `ProcessGraphics` (`:372-1060`). Os contadores CE/DE criam dependências explícitas em
   `:276-307` e `:998-1009`.
3. Child IBs são coroutines recursivas em CE (`:307-317`), graphics (`:988-998`) e compute
   (`:1126-1136`). O bit `chain` existe em `pm4_cmds.h:872`, mas call e chain atualmente retornam
   ao pai da mesma forma.
4. O tipo `Task` usa `initial_suspend` e `final_suspend` como `suspend_always` e não tem destrutor
   owner (`liverpool.h:150-180`). O task top-level é destruído pelo loop; child tasks que chegam a
   `done()` não são destruídos nos três pontos acima.
5. Compute/ASC é decodificado em `ProcessCompute` (`liverpool.cpp:1062-1334`). O read pointer é
   publicado após executar handler/child (`:1325-1330`), exceto a cópia parcial do ring em
   `:1091-1108`.
6. `ProcessCommands()` é consultado em boundaries de pacote CE, DE e ASC (`:268`, `:391`, `:1077`)
   e toma o mutex da fila de comandos (`:190-202`).
7. Draws e dispatches são classificados em `liverpool.cpp:578-805`; cada caminho consulta
   `DebugState::DumpingCurrentReg()` sob shared lock (`debug_state.h:207-210`).

### 2.3 Rasterizer e backend

Para um draw direto, `Rasterizer::Draw` executa:

1. `Scheduler::PopPendingOperations` (`vk_rasterizer.cpp:211-214`).
2. `PipelineCache::GetGraphicsPipeline` (`:221`; implementação em
   `vk_pipeline_cache.cpp:553-654`).
3. `PrepareRenderState` (`vk_rasterizer.cpp:132-169`, chamado em `:226`), que reconstrói
   `ImageDesc` e chama `TextureCache::FindImage` por target.
4. Preparação de vertex/index e recursos (`vk_rasterizer.cpp:228-236`).
5. `BeginRendering` (`:1070-1228`), descriptor writes e dynamic state.
6. `vkCmdDrawIndexed` ou `vkCmdDraw` (`:248-252`).

Os caminhos indirect e dispatch repetem o poll de completion em `vk_rasterizer.cpp:262`, `:339` e
`:369`. `PopPendingOperations` chama `MasterSemaphore::Refresh`, que consulta
`vkGetSemaphoreCounterValue` (`vk_scheduler.cpp:121-128`, `vk_master_semaphore.cpp:31-44`). O submit
ainda faz refresh e executa callbacks novamente (`vk_scheduler.cpp:151-201`).

A branch já contém otimizações relevantes que devem ser preservadas: batching de stream copies,
reuse same-tick de slices, caches de buffer/image/view, cache parcial de descriptors de imagem,
gerações de pipeline e caches de estado por command buffer. O plano estende essas estruturas em vez
de criar caminhos paralelos incompatíveis.

### 2.4 Submit, completion e apresentação

1. Cada `Scheduler` termina seu command buffer, obtém um tick de timeline e chama
   `vkQueueSubmit` sob `Scheduler::submit_mutex` (`vk_scheduler.cpp:151-200`).
2. Draw, present e flip possuem `Scheduler`/timeline próprios (`vk_presenter.cpp:496-503`). O
   `VkQueue` graphics e present é o mesmo handle (`vk_instance.cpp:561-562`).
3. EOP/EOS chama download e publica fence/IRQ durante o parse, antes de a GPU concluir
   (`liverpool.cpp:850-885`). `ReleaseMem` compute faz o equivalente em `:1302-1314`.
4. `AcquireMem` é ignorado em graphics e compute (`:956-958`, `:1183-1185`). `WaitRegMem` e
   semaphores busy-yield (`:944-986`, `:1282-1300`).
5. `TextureCache::ProcessDownloadImages` pode executar um `Scheduler::Finish` por imagem
   (`texture_cache.cpp:64-105`).
6. `PrepareFrame` ainda resolve imagem, grava FSR/PP e faz flush antes do trabalho estritamente de
   present (`vk_presenter.cpp:697-784`). `GetRenderFrame` pode bloquear em fence
   (`:1112-1148`).
7. `vkQueuePresentKHR` usa novamente o mutex global (`vk_presenter.cpp:1098-1104`), mas uploads
   ImGui fazem submit/waitIdle diretamente (`imgui_impl_vulkan.cpp:230-246`, `:879-885`).

## 3. Estados que não podem ser confundidos

**DECISÃO:** qualquer implementação deve representar separadamente:

| Estado | Significado | Autoridade |
|---|---|---|
| `parser_completed` | Pacote top-level foi copiado/validado e nenhum decoder referencia o ring | GCP/decoder |
| `sealed` | Prefixo Vulkan foi finalizado e recebeu token reservado | `Scheduler` produtor |
| `submitted` | Queue owner entregou a submissão ao driver | Vulkan queue owner |
| `gpu_completed` | Timeline/fence do token foi atingido | completion dispatcher |
| `writebacks_done` | Downloads/GDS/backing copies obrigatórios terminaram | dispatcher/data worker |
| `guest_published` | Label/fence foi escrito e IRQ/control foi publicado em ordem | GCP control lane |
| `present_completed` | Present/fence host terminou | Presenter/queue owner |
| `vblank_completed` | Evento VideoOut observável pelo guest ocorreu | VideoOut |

`SubmitDone` pode juntar parser, submit e GPU/guest publication do epoch, mas não deve esperar
vblank. Avançar read pointer em `parser_completed` não permite liberar recursos Vulkan nem publicar
fence guest.

Completion é FIFO por fila/engine guest. Uma FIFO global é proibida: um record graphics anterior
pode aguardar um ReleaseMem de ACE posterior. Ordem cross-queue só nasce de semáforo, endereço,
counter, epoch join ou outra dependência explícita.

## 4. Gargalos confirmados e prioridade

| Tier | Fato confirmado | Impacto causal | Condicionalidade |
|---|---|---|---|
| P0 | EOP/EOS/Release publica antes de GPU completion | Fence cedo, readback incorreto e necessidade de waits defensivos | Sempre que o evento é observável |
| P0 | `RenderState state;` não inicializa bytes comparados por `memcmp` | Cache miss espúrio/UB e rendering state instável | `vk_rasterizer.cpp:1074`, `vk_scheduler.h:45-55` |
| P0 | `StreamBuffer::Commit` consulta o próximo watch livre | Perde coalescing same-tick e aumenta waits | `buffer.cpp:215-240` |
| P0 | Buffer GC constrói `clean_up`, mas nunca percorre LRU | Pressão crescente, alocação/driver stalls | `buffer_cache.cpp:1403-1426` |
| P0 | Fault bitset usa load/OR/store não atômico | Bits perdidos entre invocações | `spirv_emit_context.cpp:1182-1191` |
| P0 | `submit_done`, queue count e queue size usam protocolos inconsistentes | Epoch/wake perdido ou data race | `liverpool.h:73-79`; `liverpool.cpp:225-236`, `:251-257`, `:978-980`, `:1398-1400` |
| P1 | Quatro polls de timeline por draw/dispatch | Syscall/driver query no hot path | Relevante em títulos draw-heavy |
| P1 | Download de imagem pode dar `Finish` por imagem | GCP bloqueia GPU/CPU serialmente | Recurso ativo quando linear readback está habilitado |
| P1 | Submits por par, wake/locks e scan de até 57 filas | Cache misses, mutex e branches antes de decodificar | Títulos com muitos submits/ASC |
| P1 | Coroutine/IB recursivo e semântica de chain incorreta | Alocação/state machine implícita e leaks de child handles | Títulos com IBs frequentes |
| P1 | Busy-yield de waits e semáforos | GCP gira ou rescaneia filas bloqueadas | Workloads ACE/graphics sincronizados |
| P1 | `uses_dma` percorre todas as ranges mapeadas | Trabalho proporcional ao mapa, não ao dirty set | Shaders com `ReadConst` dinâmico |
| P1 | Fault parser varre 8 MiB/32768 workgroups | GPU/driver work fixo para poucos faults | Todo submit DMA com fault processing |
| P1 | RT faz `FindImage`/mutex por target/draw | Reconstrução cold no hot path | Render target estável entre draws |
| P1 | Topology epochs globais invalidam caches não relacionados | Cascata de cold resolves | Jogos com alias/streaming de imagens |
| P1 | Descriptor cache ignora `pBufferInfo` | Push/update repetido por draw | Resource bindings estáveis |
| P1 | BDA register emite staging+barriers+copy por buffer | Driver calls e barriers fragmentadas | Buffer churn/DMA |
| P1 | Presenter ainda prepara frame no caminho GCP | Fence wait, FSR/PP e frame pool podem bloquear produtor | Flip/present frequente |
| P2 | Vertex plan e dynamic state são recalculados amplamente | CPU scalar/branches apesar de emissão suprimida | State-heavy draws |
| P2 | Pipeline miss é síncrono e warmup é serial | Stutter cold p99 | Primeira execução/novas variantes |
| P3 | Small buffers read-only são copiados a cada draw | Bandwidth CPU e ring pressure | Reuso entre ticks, baixa escrita |
| P3 | Direct multidraw/SIMD/NUMA podem ajudar | Incerto até medir runs compatíveis e hotspots | Telemetria obrigatória |

## 5. Arquitetura alvo

### 5.1 Ingestão e decoder data-oriented

Substituir a coroutine recursiva por state machines explícitas, mantendo CE e DE no mesmo
`SubmissionState`:

```cpp
struct IbFrame {
    const u32* pc;
    const u32* end;
    const u32* return_pc;
    bool owns_copy;
};

struct DecoderContext {
    const u32* pc;
    const u32* end;
    SmallVector<IbFrame, InlineIbDepth> stack;
    WaitToken blocked_on;
    u32 packet_budget;
    u8 opcode;
    bool parser_completed;
};

struct QueueHot {
    DecoderContext* active;
    u64 queue_sequence;
    u32 cs_generation;
    u16 qid;
    u8 state;
    u8 priority;
};

struct QueueCold {
    std::mutex producer_mutex;
    std::vector<u32> copied_top_level;
    std::array<u32, 1024> ring_scratch;
    DebugMarkerState debug;
};
```

- IB call empilha retorno; `chain=1` substitui o frame atual e nunca retorna.
- CE e DE têm contexts distintos; waits de counter bloqueiam somente o context dependente.
- `ready_mask`, `blocked_mask` e `active_mask` são `u64`, cobrindo as 57 filas sem scan linear.
- `curr_qid` é atualizado antes de qualquer backend call; `GetCsRegs()` depende dele
  (`liverpool.h:131-133`).
- Yield ocorre somente entre pacotes completos. Budget só força troca quando outra fila está ready.
- Scratch de 4 KiB e debug state ficam cold; hot state deve caber em poucas cache lines, com
  `static_assert` e counters de miss para validar o layout.

### 5.2 Submit batch e epochs

`SubmitGfxBatch` recebe uma mensagem MPSC contendo a sequência de inicialização e todos os pares
DCB/CCB em FIFO. Cada par preserva seu reset CE atual; não há `Flush` por elemento. Publicação gera
uma sequência monotônica e um único wake.

Substituir `bool submit_done` por:

```cpp
using QueueCut = std::array<u64, NumTotalQueues>;

struct EpochEnd {
    u64 epoch;
    QueueCut inclusive_cut;
};

std::array<std::atomic<u64>, NumTotalQueues> published_queue_sequence;
std::atomic<u64> submitted_parser_epoch;
u64 completed_parser_epoch; // GCP owner only
```

Cada `SubmitGfxBatch`, `SubmitAsc`/ding-dong e control item de fila recebe um `queue_sequence`
monotônico ao ser publicado. Um protocolo curto de publicação comum cobre incremento+enqueue; ao
publicar `EpochEnd`, ele bloqueia novas publicações somente pelo tempo de copiar atomicamente os 57
`published_queue_sequence` para `inclusive_cut` e enfileirar o marker. Assim, item
`queue_sequence <= inclusive_cut[qid]` pertence ao epoch; submit/ding-dong posterior tem sequência
maior e fica fora dele.

Ao consumir `EpochEnd`, o GCP junta cada parser somente até `inclusive_cut[qid]`, sela o work aberto
desse corte (fila sem item no corte é no-op) e cria um join observacional sobre seus
`SubmissionToken`. O join não ordena filas entre si: completion continua FIFO por guest queue e
cross-queue continua dependendo apenas de edges explícitos. `GpuIdle` é publicado somente após as
ações guest do corte; novas submissões não podem perder wake durante o reset do gate.

### 5.3 Wait tokens e fila de controle

```cpp
struct WaitToken {
    enum class Kind : u8 { None, GuestAddress, CeCounter, Semaphore, Submission, VideoOutLabel };
    Kind kind;
    u16 qid;
    u64 key;
    u64 expected;
    u64 mask;
};
```

Um wait primeiro testa a condição. Se falsa e o produtor pode estar em command buffer aberto, sela
esse produtor; depois bloqueia somente a fila no token. Completion/write notifier acorda as filas
indexadas pelo endereço/token. O GCP drena controles:

1. antes de selecionar a próxima task;
2. a cada budget limitado de pacotes;
3. antes de dormir;
4. imediatamente quando todas as filas estão blocked.

O budget impede monopolização, mas deve haver um limite máximo de latência configurável e medido.
`SendCommand<true>` participa da mesma fila de alta prioridade.

### 5.4 Tokens e queue owner

A primeira versão preserva uma timeline por `Scheduler`:

```cpp
struct SubmissionToken {
    TimelineDomain timeline_domain;
    u64 value;
    u64 enqueue_sequence;
};

struct SealedSubmission {
    SubmissionToken token;
    vk::CommandBuffer command_buffer;
    DurableSubmitInfo waits_signals;
    OptionalFence fence;
    u64 guest_queue_sequence;
    u64 recording_epoch;
    SmallVector<ResourceLease, 16> retained_resources;
    SmallVector<CompletionId, 4> completions;
};
```

O produtor executa `EndRendering`, termina o command buffer, reserva o tick, cria o token, publica
o objeto durável e aloca o próximo command buffer. O queue owner é a única thread que chama
`queue.submit/submit2/presentKHR`; ele nunca espera timeline, fence, GCP ou callback.

O owner pode drenar itens já disponíveis e emitir vários `VkSubmitInfo2` numa chamada, mantendo
waits e signals intermediários. Não espera artificialmente para formar batch. Como uma chamada
aceita um único fence, itens que exigem fences diferentes não são combinados. FIFO por guest queue
e edges explícitos cross-queue são preservados; ordem de chegada MPSC não substitui
`guest_queue_sequence`.

O completion dispatcher é outra thread. Ele nunca submete ou apresenta. Consulta/waita timelines
em batch, avança records prontos por fila e roteia ações conforme affinity.

### 5.5 CompletionRecord e publicação guest

```cpp
struct CompletionRecord {
    CompletionId id;
    std::variant<ImmediatePrerequisite, SubmissionToken> prerequisite;
    u16 guest_queue_id;
    u64 queue_sequence;
    u64 submit_epoch;
    EventScope scope;       // EOP, PS-EOS, CS-EOS, Release, flip, readback
    CacheSemantics cache;
    SmallVector<WritebackOp, 4> prerequisite_writebacks;
    Optional<FenceStore> fence_store;
    Optional<GuestIrq> irq;
    SmallVector<ResourceLease, 8> retained_resources;
    CompletionState state;
};
```

Lifecycle obrigatório:

```text
Token:     Parsed/Sealed -> Submitted -> GpuComplete -> WritebacksDone -> GuestPublished
Immediate: Parsed -------------------------------------> WritebacksDone -> GuestPublished
```

Se o evento encontra trabalho Vulkan aberto na própria fila, sela esse prefixo e usa seu token. Se
não há novos comandos, mas existe predecessor GPU na fila, usa o token desse predecessor. Se nunca
houve predecessor GPU aplicável, usa `ImmediatePrerequisite`: não cria command buffer/submit vazio e
vai direto a writeback/store/IRQ. Mesmo immediate permanece atrás de records anteriores pelo
`queue_sequence`; o atalho não viola FIFO guest nem cria ordem cross-queue.

A ordem dentro de um record é fixa: download/GDS/backing copy crítico, store de label/fence,
publicação do memory range, IRQ/control e wake de waiters. Readback correto não é cold I/O.
Screenshot, encoding, log e dump usam uma fila cold separada e nunca atrasam fence/IRQ.

`Platform::IrqC::Signal` executa subscribers sincronamente sob mutex (`src/core/platform.h:38-82`).
O dispatcher não chama o handler: publica um controle de alta prioridade e o GCP chama `Signal`
sem locks do dispatcher.

Store guest captura valor, largura, alignment e um `MappingLease`, não ponteiro de pacote ou guest
cru. `MemoryManager::TryWriteBacking` exige mapping válida (`core/memory.cpp:374-395`). Unmap
invalida logicamente a range e retarda a liberação física até leases antigos terminarem ou serem
cancelados de forma definida.

### 5.6 Regras exatas de pacotes observáveis

| Evento | Ação |
|---|---|
| EOP/EOS/Release com store, IRQ ou writeback | Se houver work aberto, selar o prefixo; senão usar predecessor ou `ImmediatePrerequisite` |
| EOP sem ação observável/cache dependency | Preservar scope/order necessário no recording; nunca criar submit vazio |
| EOS | Registrar stage scope; inicialmente completion da submissão inteira é conservadora |
| ReleaseMem GDS->memory | Gravar copy/readback antes do seal; publicar store/IRQ após writeback |
| AcquireMem | Emitir dependency/barrier de range/cache no ponto do pacote; não esperar CPU por padrão |
| WaitRegMem/semaphore true | Continuar sem seal |
| Wait false dependente de trabalho aberto | Selar o produtor, criar token e bloquear só a fila |
| WriteData/DmaData | Preservar ordem e `wr_confirm`; não converter automaticamente em EOP |
| PfpSyncMe | Preservar ordering PFP->ME; no decoder host serial não implica submit, salvo produtor aberto dependente |
| CE/DE counter | Bloquear apenas o decoder dependente, sem boundary Vulkan artificial |
| IB call/chain/return ou budget yield | Nenhum submit/invalidation por si só |
| SubmitDone | Fechar epoch, não present/vblank |
| PatchedFlip | Criar e selar snapshot pinado; trailing EOP continua record independente |

O SDK oficial confirma as diferenças: `include_common/gnm/drawcommandbuffer.h:2316-2345`
documenta EOP/cache events; `:2379` limita EOS ao stage relevante; `:2398` define o bloqueio de
`waitOnAddress`; `:2414` distingue parser stall; `:2476-2492` descreve
`waitForGraphicsWrites/flushShaderCachesAndWait`. Compute ReleaseMem está em
`include_common/gnm/dispatchcommandbuffer.h:517-534`. A implementação inicial pode ser
conservadora no escopo Vulkan, mas preserva `scope/cache` no record para refinamento posterior.

### 5.7 Presenter por snapshot pinado

```cpp
struct PinnedFrameSource {
    ImageId image_id;
    u64 image_uid;
    u64 alias_epoch;
    u64 backing_generation;
    vk::Image image;
    vk::ImageView view;
    vk::Extent2D extent;
    vk::Format format;
    SubmissionToken ready;
    ImageLease lease;
};
```

No `PatchedFlip`, o GCP resolve/atualiza a imagem, termina rendering, registra a transição, pina
imagem/view/backing e sela o source. O Presenter recebe somente o snapshot imutável, obtém frame
livre, executa FSR/PP/present e libera o pin por completion própria. Ele não chama `FindImage`, não
muta `TextureCache` e não reconsulta endereço guest que pode ter sido reutilizado.

O pool deve ser dimensionado pelo máximo realmente aceito pela fila VideoOut, incluindo
last-presented e screenshot retention. O guard atual testa `flip_pending_num > 16` antes de
incrementar e pode admitir 17 itens (`src/core/libraries/videoout/driver.cpp:292-305`); a capacidade
deve ser derivada e o guard corrigido junto do snapshot pool. No limite, não sobrescrever, descartar
silenciosamente nem bloquear o GCP; aplicar backpressure na API que aceita o flip. Buffer
label/vblank permanece evento de apresentação, separado de `SubmitDone`.

### 5.8 Gerações e fastpaths de backend

Usar gerações distintas, cada uma com origem completa:

| Geração | Incrementos obrigatórios |
|---|---|
| graphics state categories | SET_CONFIG/CONTEXT/SH/UCONFIG alterado; `ClearState`; IndexType; draw index/base/count; NumInstances; SetBase; streamout |
| per-queue compute | SET_SH memcpy, dispatch dimensions/initiator e writes diretos da queue |
| render target/depth | CB/DB regs e `last_cb_extent[]`/`last_db_extent`, mesmo sem mudança nos bytes normais |
| content range | CPU write notifier, CE DumpConstRam, WriteData, DmaData, fill/copy, occlusion/event store e completion writeback |
| directed alias | register/unregister/merge/expand de imagem e Map/Unmap que altere identidade |
| backing | troca/criação em `Image::SetBackingSamples` |
| recording | novo command buffer/seal/rollover |

`WriteGraphicsRegisters` já detecta mudanças seletivamente (`liverpool.cpp:330-370`). Ele deve
retornar uma category mask e incrementar cada categoria tocada uma vez por intervalo de registro,
não uma vez por word. O compute memcpy atual em `:1196-1209` recebe geração por queue.

Todo write de memória passa por um único `NotifyGuestMemoryWrite(range, source, content_epoch)`.
Isso inclui os opcodes diretos que não escrevem registradores. O notifier atualiza dirty ranges,
acorda waits de endereço e invalida apenas os plans/tokens dependentes.

## 6. Plano por fases e gates

### Fase 0 — telemetria causal e baseline reproduzível

**Arquivos:** novo módulo pequeno em `src/common/` ou `src/video_core/`; pontos de instrumentação em
`gnmdriver.cpp`, `liverpool.cpp`, `vk_rasterizer.cpp`, `vk_scheduler.cpp`, caches e Presenter.

**Modificações:**

- Ring fixo por thread, pré-alocado e sem mutex no caminho crítico.
- Evento compacto, por exemplo `{timestamp, type, thread, arg0, arg1}`; strings e formatação ficam
  fora do ring.
- Contadores: pacotes/opcode, bytes DCB/CCB, submits/wakes, scan de queues, ready/blocked durations,
  IB depth, draws, pipeline hit/miss, RT/descriptor/buffer token hit, bytes staging, barriers/copies,
  timeline polls, driver submit calls, submit queue depth, writeback latency e GPU idle gap.
- Dump exclusivamente no encerramento normal por `Alt+F4`; nunca arquivo/log no critical path.
- Debug markers e tracing detalhado compiláveis/desligáveis; pre-PR remove instrumentação temporária
  que não seja a telemetria aprovada.

**Gate:** baseline com dispersão inter-run calculada e decomposição PM4, prepare, driver, wait e
present. Nenhuma fase de hipótese é aceita sem contador causal.

### Fase 1 — correções isoladas de risco baixo

**Arquivos/funções:**

- `vk_rasterizer.cpp:1070-1228`: inicializar `RenderState state{}` e preencher explicitamente
  depth/stencil union antes do `memcmp`.
- `buffer.cpp:215-240`: consultar/estender `current_watches[current_watch_cursor - 1]`.
- `buffer_cache.cpp:1403-1426`: chamar
  `lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up)`. O callback retorna `true`
  quando `max_deletions == 0`; `DeleteBuffer` pode remover o item porque `ForEachItemBelow` captura
  `next` antes do callback (`common/lru_cache.h:56-74`). Centralizar a amostra de memory budget
  compartilhada com `texture_cache.cpp:1048-1110`.
- `spirv_emit_context.cpp:1182-1191`: trocar fault OR por atomic e usar o valor anterior para
  deduplicar.
- `liverpool.cpp:225-236`, `:978-980`, `:1398-1400`: eliminar leituras de
  `num_mapped_queues/submits.size` sem o protocolo de sincronização correto.
- `debug_state.h:207-210`: adicionar flag atômica hot para evitar shared lock por draw; estado cold
  continua protegido.

**Invariantes:** nenhuma nova thread, nenhum novo submit, nenhuma mudança na ordem de pacote.

**Risco/rollback:** commits separados por correção. Reverter individualmente se render hash,
watch wrap, GC download ou fault count divergir.

**Gate:** zero validation error/crash; RenderState determinístico; watches same-tick colapsam;
buffer allocations estabilizam sob pressão; fault count não perde bits em stress concorrente.

### Fase 2 — `SubmitGfxBatch`, epochs e storage ownership

**Arquivos:** `gnmdriver.cpp:2200-2324`, `liverpool.h:70-121`, `liverpool.cpp:204-262`,
`:1336-1401`.

**Modificações:**

1. Criar mensagem owned de batch com init prefix e pares DCB/CCB ordenados.
2. Copiar command storage quando a configuração exigir antes de liberar a fonte; storage é por
   submission/refcounted, nunca arena global resetada em `SubmitDone`.
3. Um lock/wake por batch; preservar reset CE por par.
4. Substituir `submit_done` por `EpochEnd {epoch,inclusive_cut[57]}` e state machine de parser
   epoch. `SubmitGfxBatch`, `SubmitAsc`/ding-dong e control messages recebem `queue_sequence` no
   mesmo protocolo curto que captura/publica o corte.
5. Separar `parser_completed` de `GpuIdle` e de presentation. `EpochEnd` junta em cada queue apenas
   itens `queue_sequence <= inclusive_cut[qid]`; trabalho posterior, vblank e present ficam fora.
6. Drenar control messages em batch nos boundaries já existentes, não tomar mutex por pacote.
7. Integrar o gate `AreSubmitsAllowed`/`submission_lock` (`gnmdriver.cpp:162-165`, `:2313-2324`)
   ao protocolo de publicação: capturar `QueueCut`, publicar `EpochEnd` e fechar/reabrir o gate sem
   janela em que item posterior entre no corte ou seja apagado pelo consumidor.

**Dependências:** telemetria da Fase 0. Prepara storage seguro para early ASC rptr e decoder
iterativo.

**Risco/rollback:** jogos podem depender de init sequence e counter reset por par. Manter modo
compatível que publica o mesmo batch mas executa exatamente a sequência anterior.

**Gate:** FIFO byte-a-byte idêntica no trace; nenhuma perda de wake em submit concurrente;
redução mensurável de locks/wakes por par; `SubmitDone` inclui todos os itens até cada corte e
nenhum SubmitAsc/ding-dong publicado depois dele.

### Fase 3 — decoder iterativo, masks e waits cooperativos

**Arquivos:** `liverpool.h:139-211`; CE `liverpool.cpp:264-328`; graphics `:372-1060`; compute
`:1062-1334`; tipos IB em `pm4_cmds.h:862-884`.

**Modificações:**

- Remover `Task` dos três decoders; introduzir `DecoderContext` e stack inline com overflow cold.
- Implementar call push e chain replace; destruir/copiar storage em `parser_completed`.
- Separar `QueueHot/QueueCold`; ready/blocked/active masks e seleção por bit scan/priority.
- Budget por pacote ativado apenas quando existe concorrente ready.
- Substituir busy-yield por `WaitToken`; testar condição antes de bloquear.
- Publicar ASC read pointer com `std::atomic_ref<u32>::store(..., release)` após copiar e validar
  todo pacote top-level; o producer deve observar o valor com acquire/coerência equivalente. IB
  child é copiado/owned antes de liberar sua fonte quando necessário.
- Tratar pacote que excede scratch por cold storage pré-alocado ou erro definido.
- Mover warnings de opcode não implementado/atípico (`liverpool.cpp:934-942`, `:1016-1024`,
  `:1212-1216`) para counters rate-limited; nenhuma formatação/log por ocorrência no hot path.
- Preservar `curr_qid`, CS state por queue e marcador debug ao alternar queues.

**Invariantes:** nunca alternar no meio do pacote; CE/DE counter order intacta; nenhum yield cria
barrier/submit; prioridade não causa starvation.

**Risco/rollback:** manter decoder coroutine atrás de opção temporária até traces de opcode/PC/IB
coincidirem. Remover o modo antigo antes de PR final.

**Gate:** traces equivalentes em IB call/chain/return; ring producer não sobrescreve packet;
zero spin quando todas as queues esperam; menor GCP CPU/queue scan sem regressão de frametime.

### Fase 4 — `SubmissionToken`, `CompletionRecord` e semântica PM4

**Arquivos:** `vk_scheduler.{h,cpp}`, `vk_master_semaphore.{h,cpp}`, `liverpool.cpp:850-986`,
`:1282-1314`, `texture_cache.cpp:64-105`, `buffer_cache.cpp:442-496`, `core/memory.{h,cpp}`,
`core/platform.h:38-82`.

**Modificações:**

1. Reservar tick e construir
   `SubmissionToken {timeline_domain,value,enqueue_sequence}` ao selar.
2. Implementar records per-queue com prerequisite
   `variant<ImmediatePrerequisite, SubmissionToken>` e os dois lifecycles completos.
3. Agregar copies de imagem/buffer/GDS no mesmo prefixo e fazer um seal, não `Finish` por imagem.
4. Implementar EOP/EOS/Release store/IRQ apenas após writeback.
5. Implementar Acquire como dependency/barrier com range/cache semantics.
6. Implementar wait-index por address/token; false wait sela produtor pendente quando necessário.
7. Criar `MappingLease`/generation e remover callbacks que capturam raw guest pointer.
8. Dispatcher publica GCP controls; GCP chama IRQ e muta caches.
9. Sampling de GPU clock/perf event ocorre na completion, não no parse.

**Semântica:** writeback crítico termina antes de fence/IRQ. Um record sem novos comandos usa o
último token predecessor quando ele existe; sem predecessor, usa `ImmediatePrerequisite` e avança
direto para writeback/publicação. Nenhum caso cria submit vazio, e ambos respeitam FIFO por guest
queue.

**Risco/rollback:** a timeline inteira é inicialmente um scope conservador para EOS. Manter scope
no record e feature flag para comparar com o caminho síncrono durante desenvolvimento.

**Gate:** testes de label/EOP/Release nunca observam store cedo; GDS/image backing está correto
antes do IRQ; wait forward-progress sem deadlock; nenhum `Finish` por imagem; callbacks não acessam
mapping reutilizada. EOP/store numa queue sem predecessor GPU publica em FIFO com zero submit
Vulkan adicional; o mesmo evento com predecessor espera exatamente o token desse predecessor.

### Fase 5 — dispatcher dedicado, fault owner-affine e remoção de polls

**Arquivos:** `vk_scheduler.cpp:121-128`, `vk_rasterizer.cpp:211-369`,
`buffer_cache/fault_manager.cpp:80-174`, caches e GCP control queue.

**Modificações:**

- Thread de dispatcher faz wait/refresh batched para todos os timeline domains.
- Fault worker só lê staging e publica resultado owned; `FindBuffer`, page table, SlotVector e LRU
  executam no GCP owner.
- GCP drena callbacks sob budget e max-latency.
- Remover os quatro `PopPendingOperations` de Draw/Dispatch e o refresh duplicado do submit.
- Retirement de staging, views, descriptors e buffers usa tokens, não polling oportunista.

**Dependência:** CompletionRecord correto. Não remover polls antes de todo callback possuir owner e
token.

**Gate:** zero `vkGetSemaphoreCounterValue` por draw; mesma latência de fence/IRQ ou menor;
dispatcher não executa queue calls/cache mutations; GCP não sofre starvation de completion.

### Fase 6 — queue owner e stream total submit/present

**Arquivos:** `vk_scheduler.cpp:151-201`, `vk_presenter.cpp:1091-1104`, `vk_swapchain.cpp:82-150`,
`imgui_impl_vulkan.cpp:230-255`, `:703-713`, `:879-885`, `imgui_core.cpp:152-159`.

**Modificações:**

1. Adicionar MPSC de `SealedSubmission` e requests de present/control.
2. Queue owner exclusivo para submit/present; produtor continua finalizando/alocando CB.
3. Drenar itens disponíveis em `vkQueueSubmit2` com vários infos e signals intermediários.
4. Converter submit/waitIdle ImGui em submissão/retirement coordenado. A chamada
   `ImGui::Core::TextureManager::Submit()` hoje feita no produtor em `vk_scheduler.cpp:192` vira
   metadado/lease do `SealedSubmission`; liberações só ocorrem depois do token correspondente.
5. Swapchain recreate/HDR/device idle vira request cold síncrono ao owner após drain.
6. `presentKHR` devolve resultado owned ao Presenter; out-of-date/suboptimal agenda recreate depois
   do drain sem segurar locks de cache/frame. O frame só volta ao free pool sob seu fence/token.
7. Device loss define estado terminal, rejeita novos itens e acorda todos os waiters com erro;
   remover loop infinito de `MasterSemaphore::Wait` (`vk_master_semaphore.cpp:58-67`).
8. Shutdown: parar produtores, selar, drenar owner, drenar dispatcher/owner callbacks, parar threads,
   destruir recursos/device.

**Invariantes:** owner nunca espera GPU/GCP; dispatcher nunca submete/presenta; não atrasar submit
para formar batch; mesmas dependências e timelines por `Scheduler`.

**Gate:** todas as queue calls passam pelo owner; ThreadSanitizer/validation não aponta host access
concorrente; driver submit calls caem quando há burst; enqueue-to-submit não aumenta além do noise
floor; out-of-date/suboptimal e shutdown não deadlockam.

### Fase 7 — Presenter por snapshot

**Arquivos:** marker `src/video_core/amdgpu/liverpool.cpp:419-423`; patch do label/EOP em
`src/core/libraries/gnmdriver/gnmdriver.cpp:2120-2164`; VideoOut em
`src/core/libraries/videoout/video_out.cpp:345-359` e
`src/core/libraries/videoout/driver.cpp:292-360`; `vk_presenter.cpp:697-784`, `:860-1110`,
`:1112-1148`; TextureCache/Image.

**Modificações:**

- Criar `PinnedFrameSource` no GCP no marker e selar o prefixo que produz a imagem.
- Mover `GetRenderFrame`, FSR, PP, screenshots e present para Presenter.
- Retenção física por token/fence do Presenter; logical image identity permanece owner GCP.
- Dimensionar pool pelo limite aceito e corrigir o guard da fila; backpressure na aceitação.
- Separar render-ready, present-done e vblank/label.

**Risco/rollback:** maior risco é reutilização de backing enquanto Presenter lê. Manter pin count e
assert `{id,uid,backing_generation}` até completion; opção temporária de preparar frame no GCP para
comparação.

**Gate:** GCP nunca espera `GetRenderFrame`; nenhuma lookup/mutação TextureCache no Presenter;
source não muda durante FSR/PP; ordem flip/EOP/label coincide com baseline.

### Fase 8 — DMA fault queue compacta e BDA dirty batching

**Arquivos:** `buffer_cache.h:42-46`, `fault_manager.cpp:16-174`,
`fault_buffer_process.comp:20-34`, `spirv_emit_context.cpp:1152-1200`,
`vk_rasterizer.cpp:430-462`, `buffer_cache.cpp:1157-1201`, `:1336-1400`.

**Modificações:**

1. Layout fault: bitset de dedup, counter, overflow e array compacto de page indices `u32`.
2. Guest shader usa `atomicOr`; só o primeiro lane que vê bit ausente executa `atomicAdd`.
3. Parser dispatcha pela capacidade compacta, não por todas as palavras do bitset.
4. Overflow nunca apaga fault não emitido: full-scan/chunk fallback conserva bits restantes e
   agenda novos drains até vazio.
5. Remover log por página do callback.
6. Manter areas de download em free-list por token; nunca `scheduler.Wait` para reciclar slot.
7. Manter conjunto de buffers DMA dirty/registered para sincronizar apenas ranges tocadas. Não
   eliminar eager residency apoiando-se em fault: o primeiro dynamic read já recebeu fallback e
   não há replay seguro.
8. `Register/Unregister` atualiza page table CPU logicamente no GCP e acumula dirty page ranges.
9. `UploadPlan` emite staging/copies BDA agrupados e uma barrier para o consumidor DMA.

**Gate:** fault stress sem bit perdido/overflow silencioso; workgroups proporcionais à capacidade,
não ao address space; zero wait de area no GCP; BDA mappings corretos antes do primeiro consumidor;
redução de ranges varridas e barriers/copies.

### Fase 9 — RT tokens, alias dirigido e Map/Unmap owner-routed

**Arquivos:** `liverpool.cpp:330-370`, `:478-535`; `vk_rasterizer.cpp:132-169`,
`:872-1228`; `texture_cache.cpp:507-667`, `:700-790`, `:904-937`, `image.cpp:790+`;
`core/memory.cpp:640-675`, `:1076-1098`; `vk_rasterizer.cpp:1363-1378`.

**Modificações:**

- Geração por color target e depth target, incluindo os hints NOP derivados.
- Token `{ImageId,uid,alias_epoch,backing_generation,image,view,resolved_desc}`.
- Register/unregister coleta IDs únicos de todas as page buckets sobrepostas e incrementa uma vez o
  `alias_epoch` de cada imagem afetada.
- Backing swap incrementa `backing_generation`.
- Hit executa `MarkTargetUsed`; slow branch permanece para `Dirty`, `needs_rebind`, sample/backing
  change, meta change ou alias mismatch.
- Remover `UpdateImage` duplicado entre Rasterizer e `PrepareRenderTarget`; registrar CMASK/FMASK/
  HTILE/stencil apenas quando a geração relevante muda.
- Manter topology epoch global apenas no exact/cold cache inicialmente.
- Roteiar Map/Unmap lógico inteiro por `Liverpool::SendCommand<true>`; physical destruction e
  mapping backing usam leases/tokens.

**Gate:** aumento alto e estável do RT token hit rate; zero lookup/mutex no hit; aliases e remap
produzem as mesmas imagens/views e render hashes; nenhum validate-then-unmap race.

### Fase 10 — ResourceLayoutPlan, descriptors, vertex e dynamic state

**Arquivos:** `vk_graphics_pipeline.cpp:444-509`, `vk_compute_pipeline.cpp:34-103`,
`vk_pipeline_common.cpp:22-63`, `vk_rasterizer.cpp:430-534`, `:697-1047`,
`buffer_cache.cpp:498-760`, `vk_scheduler.cpp:229-386`.

**Modificações:**

- Criar `ResourceLayoutPlan` imutável por pipeline final: slots, descriptor type/count/stage,
  origem, alignment, push-data offset e variant key para `NumBindings(tsharp)`.
- Construir `ResolvedResourceKey` por geração de user data, content range, alias e compute queue.
- Deep cache inclui buffers `{handle,offset,range,type}` e imagens
  `{view,layout,sampler,type}`; push descriptor cache é por command buffer/bind point/layout.
- Centralizar todas as chamadas `pushDescriptorSetKHR` para atualizar recording epoch.
- Descriptor sets não-push ficam immutable/retidos até o token; pool reset é timeline-safe.
- Criar `VertexBindingPlan` por pipeline/fetch shader; resolver somente base addresses/content
  generations por draw.
- Dynamic state usa dirty groups derivados da category mask; viewport/scissor loops só percorrem
  grupos alterados.
- `FastDrawState` inclui pipeline, recording epoch, RT tokens, resource generation fingerprint,
  descriptor fingerprint, vertex plan e dynamic dirty mask.

**Risco/rollback:** `NumBindings`, mip fallback e descriptors arrays são variantes reais, não podem
ser fixados pelo preload default. Cada plan/cache tem contador de reason-specific miss e opção de
desligar separadamente durante desenvolvimento.

**Gate:** menos `ImageDesc` copies, descriptor writes, buffer lookups, vertex recalcs e dynamic
commands por draw; zero descriptor lifetime/validation error; mesma saída em mip fallback/storage.

### Fase 11 — UploadPlan e barriers por range

**Arquivos:** `buffer_cache.cpp:1204-1400`, `buffer.h:133-154`, `image.cpp:221-399`,
`vk_pipeline_common.cpp:22-36`, memory tracker e TextureCache alias sync.

```cpp
struct ResourceUsage {
    ResourceId resource;
    u64 offset;
    u64 size;
    vk::PipelineStageFlags2 stage;
    vk::AccessFlags2 access;
    UsageKind kind;
};

struct UploadPlan {
    SmallVector<ResourceUsage, 32> usages;
    SmallVector<BufferCopy, 16> copies;
    SmallVector<BufferBarrier, 16> pre;
    SmallVector<BufferBarrier, 16> post;
};
```

**Modificações:**

- Coletar usages por draw/basic block e ordenar/agrupar em vectors inline.
- Unir reads sobrepostos e stages/access; write cria boundary e nunca é movido sobre consumidor.
- Estado de buffer por poucos intervalos adjacentes; não usar árvore/map alocado no hot path.
- Fase única: pre barriers, copies agrupadas por src/dst, post barriers para a união de consumidores.
- Dirty state é transacional: só limpar depois de snapshot em staging e comando gravado.
- Snapshot guest ocorre no GCP sob mapping válida; queue owner recebe somente CB finalizado.
- Layout image pre-copy e consumer transition continuam separados; alias buffer/image mantém os
  copies/coherence points necessários.
- Acquire/EOP/Release/Wait e operação observável encerram o horizonte de batching.

**Gate:** menos barrier/copy commands e `eAllCommands`; nenhuma hazard de range/layout sob Vulkan
validation; uploads não cruzam packet boundary; custo CPU do interval tracker menor que a economia.

### Fase 12 — persistent small buffers e pipeline cold path

**Small buffers:** manter por range
`{guest_range,mapping_generation,allocation_generation,bind_ticks,cpu_write_ticks,last_use_token}`.
Uma `PromotionPolicy` configurável contém `min_stable_bind_ticks`, `observation_window`,
`max_write_ticks` e budget; seus valores são escolhidos pelos histogramas da Fase 0, permanecendo
desabilitados por padrão até esse gate. Promover apenas UBO/VB/IB read-only que exceda a estabilidade
e fique abaixo do write limit. Criar suballocator device-local próprio, generation e dirty-page
tracking. Dirty range entra no UploadPlan antes do próximo bind. Demotion é lógica imediata e free
físico após token. Storage, DMA e BDA continuam no BufferCache canônico até existir protocolo de
alias completo. O reuse same-tick continua sendo o primeiro nível.

**Pipeline cold:** separar lookup/dependency plan do build; predecodificar chave quando gerações de
shader/state mudam e preaquecer variantes previsíveis fora do GCP. Pipeline creation async usa
inputs owned e sincronização correta do `VkPipelineCache`; o draw só espera um future quando a
pipeline é realmente necessária. Warmup/serialization atual em `vk_pipeline_cache.cpp:297-371` e
misses síncronos em `:627-654`, `:928-958` são medidos separadamente de steady state.

**Gate:** promoção só permanece se reduzir bytes copiados/ring waits e melhorar frametime acima do
noise floor sem elevar page-fault CPU; pipeline phase reduz stutter cold p95/p99 e tempo GCP sem
regredir startup/memória de forma não explicada.

## 7. DAG de dependências

```text
Telemetria
   |
   +--> correções isoladas
   |
   +--> SubmitGfxBatch/epochs/storage --> decoder/masks/early-rptr
                                      |
                                      +--> SubmissionToken/CompletionRecord/waits
                                                |
                                                +--> dispatcher/remove polls/fault affinity
                                                |          |
                                                |          +--> DMA compact/BDA batching
                                                |
                                                +--> queue owner --> Presenter snapshot
                                                |
                                                +--> mapping leases --> RT alias tokens
                                                                          |
                                     generations/content notifier --------+--> ResourceLayoutPlan
                                                                                  |
                                                                                  +--> UploadPlan
                                                                                           |
                                                                                           +--> persistent small buffers

Pipeline cold prefetch depende de gerações/storage owned, mas não de small buffers.
Multidraw depende de toda a telemetria de FastDrawState e permanece fora do caminho crítico inicial.
```

## 8. Análise DOP, caches, SIMD, batching e NUMA

### L1 e false sharing

- `QueueHot`, ready masks, counters e decoder PC ficam contíguos; mutex, vectors, 4 KiB scratch e
  debug strings ficam cold.
- Índices producer/consumer de MPSC/SPSC ocupam cache lines separadas (`alignas(64)`).
- Completion hot header fica separado de vectors/leases cold; dispatcher toca primeiro apenas
  `{domain,value,qid,sequence,state}`.
- Plans imutáveis são arrays compactos percorridos linearmente; scratch de resolução é reutilizado
  por draw, sem alocação.

### L2/L3 e coerência

- Evitar scan das 57 queues, do mapa inteiro DMA, do bitset inteiro e de todos os descriptors.
- `Regs` ocupa cerca de 208 KiB (`regs.h:20-164`); não copiar/zerar/hashear globalmente no draw.
  Usar category generation e acessar somente ranges relevantes.
- O `memset` amplo de defaults em `regs.cpp:26-49` permanece cold/baixa prioridade. Só substituir
  por template copy, dirty-page reset ou reset parcial se a telemetria mostrar `ClearState`
  frequente e custo acima do noise floor; a geração de todas as categorias continua obrigatória.
- `ImageDesc`/mip layout são cold; tokens hot carregam IDs, handles e generations.
- Queue owner e GCP trocam objetos pequenos owned, não ponteiros para vectors temporários.

### SIMD

- O decoder é branch-heavy; SIMD manual não é primeira prioridade.
- `std::memcmp`/`memcpy` já podem usar implementações vetorizadas. Instrumentar tamanho e frequência
  antes de AVX2 específico em register compare, snapshot ou descriptor fingerprint.
- `CopySparseMemoryBatch` e non-temporal copy já são uma base melhor para grandes ranges; ajustar
  threshold pela telemetria de tamanho/alignment/cache pollution.
- Opcional depois da compact queue: deduplicar faults por subgroup antes do `atomicOr` se o perfil
  mostrar contenção de waves na mesma página.
- Toda versão SIMD precisa de fallback e equivalência byte-a-byte; ganho deve aparecer em cycles,
  cache misses e frametime, não microbenchmark isolado.

### Batching

Prioridade: SubmitGfx, control messages, timeline refresh, queue submits, downloads, BDA writes,
stream copies, descriptors e barriers. Batching nunca atravessa EOP/EOS/Release observável,
Acquire, false Wait dependente, PatchedFlip snapshot, SubmitDone ou sync readback.

### NUMA e afinidade

Ryzen 5 5600 é um sistema de um NUMA node; hard pinning não oferece locality NUMA e pode prejudicar
o scheduler. Detectar topology em runtime e registrar node/processor group na telemetria. No alvo:

- preferir estruturas thread-local/SPSC e reduzir compartilhamento;
- permitir afinidade opcional apenas após medir migrações e cache misses;
- manter queue owner e GCP próximos somente se isso não competir pelo mesmo core lógico;
- colocar cold I/O fora dos hot cores.

Em hosts multi-node, alocar filas/staging no node do produtor/consumer dominante e medir remote
access. Não introduzir política NUMA obrigatória nesta série.

## 9. Matriz de efeitos laterais

| Mudança | Consumidores distantes | Risco | Proteção |
|---|---|---|---|
| Epochs/SubmitGfxBatch | GpuIdle, gate de submits, capture, CE reset | incluir trabalho posterior/perder wake | `QueueCut[57]` atômico e trace FIFO |
| Decoder stack/chain | CE/DE, compute IB, markers, copied buffers | PC/return incorreto | trace opcode/PC/depth e modo compatível |
| Early ASC rptr | ring producer e child IB lifetime | overwrite do pacote | owned packet + release store |
| Completion async | MemoryManager, IRQ, waits, cache dirty | fence antes do writeback | lifecycle e publicação estrita |
| Per-queue completion | gfx/ACE/SubmitDone | deadlock cross-queue | dependency graph explícito e epoch join |
| Queue owner | Scheduler pools, ImGui, swapchain, Tracy | lifetime ou submit reorder | sealed object owned e sequence |
| Presenter snapshot | TextureCache, Image backing, VideoOut pool | use-after-free/reuse | UID/backing generation e leases |
| Directed alias | RT/texture caches, Map/Unmap | token stale | bump de todos overlaps deduplicados |
| Resource plan | mip fallback, arrays, push descriptors | shape/lifetime incorreta | variant key e recording epoch |
| Upload range state | MemoryTracker, image alias, DMA | barrier insuficiente | transactional dirty e validation |
| Buffer GC ativado | GPU-modified download e BDA page table | apagar dado vivo | LRU age, token retirement, writeback |
| Persistent small | page tracker, budget, aliases | copy stale/churn | read-only first, promotion telemetry |

## 10. Benchmark, correção e gates quantitativos

### Protocolo

- Mesma build, configuração, save, cena, resolução, driver, clock/power mode e background load.
- Separar cold run de steady state; registrar warmup e janela medida.
- Pelo menos cinco execuções independentes por cena e frames suficientes para estabilizar o
  intervalo de confiança; aumentar a amostra quando p99 não estabilizar.
- Reportar p50/p95/p99 do CPU frametime, 1% low, GCP busy/blocked CPU, GPU idle gaps,
  enqueue-to-submit, submit-to-complete, driver calls/frame, polls/frame, bytes/copies/barriers e
  cache hit/miss reasons.
- Usar bootstrap 95% CI e dispersão inter-run. Uma fase de performance só passa se a melhora causal
  tiver CI que não cruza zero e magnitude maior que duas vezes o noise floor observado. Regressões
  maiores que o noise floor em qualquer cena exigem explicação ou rollback.
- Não aceitar ganho de GCP CPU que aumente GPU idle gap ou p99 end-to-end.

### Matriz mínima

| Workload | Estressa | Oráculos de correção |
|---|---|---|
| Draw-heavy, pipeline quente | decoder, RT, descriptor, dynamic, vertex plan | render hash/captura e draw counters |
| IB/CE-DE intensivo | call/chain, counters, budget | trace PC/opcode e ausência de hang |
| ACE + graphics waits | per-queue completion e WaitRegMem | ordem de labels/IRQs e forward progress |
| ReadConst/DMA dinâmico | BDA, faults, dirty ranges | fault pages, constant output e overflow |
| RT/texture alias + mip | directed epoch/views/layout | render hash e Vulkan validation |
| Map/Unmap/remap stress | mapping leases e cache owner | zero stale write/use-after-unmap |
| EOP/EOS/Release/GDS | writeback lifecycle | backing bytes antes de fence/IRQ |
| Flip rápido + resize/HDR | snapshot, pool, queue owner | ordem flip/vblank, sem drop/deadlock |
| VRAM pressure | GC e retirement | memória estabiliza e dados preservados |
| Pipeline cold | compile/prefetch | first-run p95/p99 e cache correctness |

### Gates específicos de resultado

- Per-draw timeline polls: exatamente zero após a Fase 5.
- Synchronous `Finish` por imagem em EOP: exatamente zero após a Fase 4.
- Queue calls fora do owner: exatamente zero após a Fase 6.
- Fault lost/overflow silencioso: exatamente zero.
- Writeback publicado depois do fence/IRQ: exatamente zero violações.
- Vulkan validation errors, render mismatches não explicados, deadlocks e use-after-free: zero.
- Cada fase P1 precisa reduzir seu contador causal e melhorar pelo menos uma métrica end-to-end
  acima do noise floor; reduzir somente nanossegundos de microbenchmark não é suficiente.

## 11. Propostas rejeitadas ou adiadas

- **Timeline global imediata:** adiada. Tokens mantêm `timeline_domain`; unificação futura é possível
  após estabilizar queue owner/completion.
- **FIFO global de completion:** rejeitada por risco de deadlock gfx/ACE.
- **Flush em todo yield/wait:** rejeitado; elimina o benefício do scheduler cooperativo.
- **Remover barriers em massa:** rejeitado; somente coalescer/estreitar com range hazard provado.
- **Fault queue como demand paging/replay:** rejeitada; primeiro acesso já executou fallback.
- **Async host compute/backend paralelo amplo:** rejeitado nesta fase; caches são owner-affine.
- **Presenter reconsultando guest address:** rejeitado; somente snapshot pinado é seguro.
- **Direct multidraw cego:** adiado. O código já usa uma chamada para indirect-multi
  (`liverpool.cpp:662+`, `vk_rasterizer.cpp:313-330`). Direct runs só avançam se a telemetria provar
  mesma pipeline, attachments, descriptors, push constants, VB/IB, dynamic state, generations,
  ausência de eventos/writes/queries/predication e semântica de DrawID preservável.
- **SIMD manual antes do profile:** adiado; decoder branch-heavy e libc/compilador já vetorizam
  operações básicas.
- **Hard NUMA/pinning no Ryzen 5600:** rejeitado.
- **Remover topology epoch global de uma vez:** rejeitado; primeiro sair do hot path e manter cold
  fallback até validar directed epochs.
- **Persistent small buffers para storage/DMA:** adiado até alias/update/retirement estarem provados.
- **“Migrar para dynamic rendering” como otimização:** rejeitado como premissa; o backend já usa
  `beginRendering`. O custo real é preparação/redundância, não a API de render pass.
- **EOS stage-exact na primeira versão:** adiado. Full-submission completion é conservadora e
  correta; scope permanece no record para recuperar overlap depois.

## 12. Ordem final de implementação

- [ ] Atualizar upstream local, comparar branch e revalidar anchors.
- [ ] Capturar baseline e noise floor com rings in-memory.
- [ ] Corrigir RenderState, stream watch, Buffer GC, fault atomic e races isoladas.
- [ ] Implementar `SubmitGfxBatch`, owned storage e parser epochs com `QueueCut[57]` inclusivo.
- [ ] Trocar coroutines pelo decoder iterativo e corrigir IB chain/call.
- [ ] Introduzir ready/blocked masks, wait tokens, budgets e early ASC rptr.
- [ ] Introduzir `SubmissionToken` por timeline domain e `CompletionRecord` per-queue com
  prerequisite token/immediate.
- [ ] Corrigir EOP/EOS/Release/Acquire/Wait e agregar readbacks antes de fence/IRQ.
- [ ] Criar dispatcher, owner callbacks e remover polling por draw.
- [ ] Criar queue owner, migrar submit/present/ImGui e fechar failure/shutdown.
- [ ] Mover present pesado para snapshot pinado.
- [ ] Compactar fault queue e agrupar BDA dirty uploads.
- [ ] Criar memory write notifier, gerações completas e Map/Unmap owner-routed.
- [ ] Criar RT tokens/directed alias e remover resolves duplicados do hot path.
- [ ] Criar ResourceLayoutPlan, deep descriptor cache, VertexBindingPlan e dynamic dirty groups.
- [ ] Criar UploadPlan/barriers por range.
- [ ] Avaliar persistent small buffers e pipeline cold prefetch.
- [ ] Avaliar multidraw/SIMD/NUMA somente pelos gates de telemetria.
- [ ] Antes de PR: revisar diff integral, remover paths alternativos/telemetria temporária, executar
  build full do projeto, clang-format/REUSE/testes exigidos e repetir a matriz completa.

O checkpoint entre fases deve ser um commit local pequeno e reversível, acompanhado dos números
baseline/novo e do gate correspondente. Nenhuma fase dependente avança se a precedente apenas
deslocar o stall para outra thread ou aumentar GPU idle gap.
