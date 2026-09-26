# Plano: readback lazy generalizado e menos sincronização, sem regressão

Estado: proposta. Nada foi implementado ainda.
Base: branch `codex/eltutz-gow` @ `26dc6a7`.

**Objetivo:** todo jogo com *Readback Linear Images* ligado passa a pagar aproximadamente o custo do
readback desligado, mantendo a correção do readback ligado. Com a opção desligada, nenhum caminho
muda.

**Restrição:** nenhuma fase pode regredir desempenho em nenhum jogo. Toda decisão tem um fallback
que é exatamente o comportamento de hoje. Não existe gate por título, formato, tamanho ou
assinatura de pacote. Não existe flag de runtime; comparações A/B são feitas entre builds.

---

## 0. Resumo

A lentidão vem de três custos independentes:

| # | Custo hoje | Quem paga | Correção | Fase |
|---|---|---|---|---|
| 1 | O CP espera a GPU em `WAIT_REG_MEM`, porque o label foi adiado até o readback cair na RAM | jogos em que a GPU espera um label de um scope com readback (o caso do GoW3) | todo readback elegível vira authority (os dados ficam na GPU e as leituras são interceptadas). O sinal passa a ser publicado quando o CP o processa | F1 |
| 2 | A CPU do guest espera EOP/EOS adiados mesmo quando nunca lê os dados (pacing de frame) | todo jogo com readback ligado que escreve RT linear ou storage image em compute | a mesma regra: o sinal só é adiado se algum dado em voo tiver consumidor na CPU | F1 |
| 3 | A GPU drena o pipeline e quebra o render pass a cada pacote de cache do guest (`EVENT_WRITE`, `ACQUIRE_MEM`), a cada cópia de readback e em barriers `ALL_COMMANDS` | todos os jogos; os readbacks, só com a opção ligada | barriers dirigidas por recurso ("epoch de flush"), readback sem cópia (direct authority), barriers estreitas e em lote, e uma política explícita para alimentar a GPU | F3–F5 |

Page faults são tratados em F2. O emulador passa a ler a memória guest sempre pela *backing view*.
Faults do guest viram instrução decodificada e emulada. Espera acontece apenas quando os bytes
pedidos ainda não existem.

---

## 1. Onde o tempo vai hoje (verificado no código)

### 1.1 Espera do CP: o custo que o fast path do GoW3 removeu

1. Um draw ou dispatch escreve uma imagem linear.
   `ScheduleRenderTargetDownload`/`ScheduleComputeDownload` registra um download pendente
   (`texture_cache.cpp:2158-2306`).
2. No EOS/EOP/ReleaseMem, `ProcessDownloadImages` faz três coisas: chama `EndRendering`, transita a
   imagem para `TransferSrc` e grava `copyImageToBuffer`. A escrita na RAM fica para a conclusão da
   GPU (`texture_cache.cpp:679`, `1150-1160`, `1270`).
3. O sinal (label + IRQ) é adiado até a conclusão física (`liverpool.cpp:1784-1794`, `1963-1973`,
   `3379-3394`). No ReleaseMem ainda há um submit extra (`WritebackReleaseMem`).
4. Um `WAIT_REG_MEM` sobre esse label encontra o valor antigo e faz duas coisas: flush e spin até a
   GPU terminar (`liverpool.cpp:3003-3044`). Essa é a espera no host. No GoW3 ela custava em média
   ~4 ms, com p95 perto de um frame inteiro (`plano.md` anterior).

O fast path atual só pula o passo 4 quando todas estas condições valem: título CUSA01715, RT 1x1
R16G16Sfloat de pitch 128 e 512 bytes, exatamente um download pendente, EOS com valor 1, e wait
Equal/≥ com ref 1 e máscara cheia. O restante continua legado. No dump v7, 90% dos readbacks do
próprio GoW3 eram storage images de compute no EOS, e esses ficam de fora hoje. A generalização
deve beneficiar o GoW3 também.

### 1.2 Espera da CPU do guest

Com readback ligado, todo EOP/EOS cujo scope tem download pendente publica label e IRQ apenas na
conclusão da GPU. A thread do guest que faz pacing esperando esse EOP (flip, fim de frame) fica
presa à latência da GPU, mesmo que nunca leia os dados. Com readback desligado, os mesmos sinais
são publicados quando o CP os processa (`liverpool.cpp:1974-1975`). Essa diferença é custo puro nos
jogos que nunca leem os dados.

### 1.3 Sincronização do lado da GPU

| Local | O que emite | Origem | Observação |
|---|---|---|---|
| `EVENT_WRITE` de flush de cache (CacheFlush*, Cs/Ps/VsPartialFlush, FlushAndInv*) → `Rasterizer::FlushCaches` (`vk_rasterizer.cpp:672-752`) | `EndRendering` + memory barrier global | fork (`c2970ce`); no upstream isso não emitia nada | frequente em Gnm |
| `ACQUIRE_MEM` → `Rasterizer::AcquireMemory` (`vk_rasterizer.cpp:584-669`) | `EndRendering` + memory barrier global | fork (`c2970ce`); no upstream era no-op | frequente em Gnm |
| drain de readback (`texture_cache.cpp:679`) | `EndRendering` incondicional, mesmo quando nada é copiado | fork | |
| cópia de readback (`texture_cache.cpp:1150-1160`) | 2 transições de layout + transfer, por imagem e por scope | upstream/fork | acontece mesmo quando ninguém lê os bytes |
| shadow serve (`buffer_cache.cpp:1902-1928`) | `EndRendering` + 2 memory barriers globais `ALL_COMMANDS` | fork | |
| `SynchronizeBuffer` (`buffer_cache.cpp:2250-2291`) | barrier por buffer inteiro, `ALL_COMMANDS`, antes e depois de cada upload | upstream | |
| qualquer buffer barrier de draw (`vk_pipeline_common.cpp:53-64`) | `EndRendering` | upstream | |
| submits `WaitProgress` (7 sites) e `WritebackReleaseMem` | submit no meio do frame | fork | ~11 submits/frame no GoW3 |
| EOS `GdsStore` (`liverpool.cpp:1797-1808`) | `Finish()`: submit + espera total da GPU | upstream | |
| readback de buffer por fault (`buffer_cache.cpp:1127`, `1216`) | `SendCommand<true>` + `Finish()` | upstream | só no modo Precise |

**Por que as barriers globais do fork existem:** `Buffer::GetBarrier` (`buffer.h:147-152`) não
emite barrier quando o acesso anterior e o novo têm a mesma máscara. Dois dispatches que usam o
mesmo buffer em RW, portanto, não se sincronizam pelo rastreio por recurso. O `CS_PARTIAL_FLUSH` ou
o `ACQUIRE_MEM` do guest era o que cobria esse buraco. Então elas não podem simplesmente sair. Elas
precisam virar barriers dirigidas (§5.1).

### 1.4 Como ler "GPU a 100%"

O contador de uso da GPU mede tempo com trabalho na fila. Uma espera do CP no host deixa a GPU
**ociosa**: aparece como uso menor e FPS menor. É isso que o GoW3 tinha antes do fast path. Já com
a GPU a 100%, o gargalo está no lado da GPU. Uma parte é trabalho útil. Outra parte é o custo das
syncs **dentro** da GPU:

- drenagem de pipeline em cada barrier;
- store/load de attachments e (des)compressão a cada quebra de render pass;
- transições de layout;
- cópias extras (readback, detile).

As duas coisas precisam cair. §1.1–1.2 atacam o primeiro caso e §1.3 o segundo. A medição usa os
intervalos GPU `DependencyDelay` que a telemetria já grava em torno das barriers, e não apenas o
contador de uso.

### 1.5 Por que o plano anterior (authority em tudo) perderia desempenho

| Risco | Efeito | Como este plano evita |
|---|---|---|
| Proteção de página de longa duração, com materialização por página inteira | a CPU mexe num objeto vizinho da mesma página e espera a GPU vários ms (o mesmo problema que motivou `e03d11c` nos workers) | F2: acesso byte-preciso. Bytes que não são da authority nunca esperam |
| Publicar o label cedo para quem lê na CPU | um padrão "lê se já estiver pronto" vira leitura bloqueante | classificação por consumidor: um range lido pela CPU volta ao comportamento legado exato |
| A cópia shadow no EOS/EOP continua quebrando o render pass | o custo de GPU fica igual ao legado | F4: direct authority, sem cópia no scope |
| Remover o flush de progresso junto com a espera | a GPU fica ociosa enquanto o CP grava o resto do frame | F1: política de kick explícita, separada da semântica de sync |

---

## 2. Princípios

1. **Cópia é barata; sincronização é cara.** A memória não é unificada. Sempre vale trocar um ponto
   de sync por uma cópia, por um adiamento ou por interceptação.
2. **Espera só existe quando um observador precisa de bytes que ainda não existem.** Mesmo nesse
   caso, espera-se apenas o tick que produz esses bytes, no ponto mais tarde possível.
3. **A representação segue o consumidor.** Dado consumido só pela GPU nunca vai à RAM. Dado
   consumido pela CPU é escrito na RAM na conclusão, sem fault e sem espera no CP.
4. **O emulador nunca gera fault na memória guest.** Leituras e escritas internas usam a backing
   view com precisão de byte.
5. **Barrier só quando um recurso escrito é consumido depois de um flush do guest.** Nunca uma
   barrier global por pacote.
6. **Toda promoção é estrutural e tem fallback legado.** Nada depende de título, formato, tamanho ou
   do valor do pacote.

---

## 3. O critério geral: por que o GoW3 funciona e quando outro caso também funciona

Hoje o emulador já publica no tempo do CP todo sinal que não tem readback no seu scope. Isso é
seguro porque a GPU trabalha com cópias próprias dos inputs (snapshot na gravação) e porque ninguém
na CPU lê os outputs sem interceptação. O único motivo para atrasar um sinal é: **ele protege bytes
que só chegam à RAM na conclusão e que podem ser lidos sem passar pelo emulador**.

Há três classes de observador depois de um sinal ou wait:

1. **Trabalho GPU gravado depois:** já está ordenado pela fila Vulkan única e pelas barriers por
   recurso. Consumidor de buffer: `SynchronizeBufferFromImage` ou shadow. Consumidor de imagem: a
   própria imagem.
2. **Leitura dos bytes no host** (CPU do guest, CP, uploads, HLE): é segura se estiver
   **interceptada**. A authority faz isso: protege as páginas, serve pelo shadow e materializa sob
   demanda.
3. **Quem observa labels e IRQs:** vê a mesma ordem de programa do CP.

**Regra:** se todos os bytes em voo até um sinal estão sob authority (interceptados), o sinal segue
a regra geral e é publicado no tempo do CP. Se existe um writeback eager em voo (dado com
consumidor na CPU, ou não elegível), o sinal espera esse commit, exatamente como no legado.

O GoW3 passava na regra porque a imagem é linear e tem um só mip. Assim o shadow tem o layout exato
da RAM, e nada eager estava em voo. Os 512 bytes, o formato R16G16, o ref 1 e o valor 1 eram
incidentais.

**Elegibilidade estrutural para authority** (a mesma que o legado precisaria para estar correto):

- imagem viva e com `GpuModified`;
- `SafeToDownload`;
- não tiled;
- 1 mip;
- endereço alinhado a 4 bytes;
- download ≤ `guest_size`;
- cabe no ring de download;
- estado de alias consistente (`TrackImageReadback` já valida quase tudo isso).

Tiled, multi-mip ou alias ambíguo seguem no legado, que continua funcionando como hoje.

---

## 4. Arquitetura do readback

### 4.1 Modos por range

| Modo | Quando | Cópia no scope | RAM | Proteção de página | Sinal |
|---|---|---|---|---|---|
| **Direct** (F4) | consumidor só na GPU; a imagem sobrevive | nenhuma | nunca, ou na retirada | read-watch | tempo do CP |
| **Shadow** (F1) | consumidor só na GPU; a imagem pode mudar antes do consumo | image→buffer pinado | sob demanda (lazy) | read-watch | tempo do CP |
| **CpuCommit** | range com consumidor na CPU aprendido, ou página "quente" | image→buffer | commit na conclusão, na thread de prioridade, sem espera no CP | nenhuma (igual ao legado) | adiado até o commit (legado exato) |
| **Legacy** | não elegível | igual a hoje | igual a hoje | igual a hoje | igual a hoje |

Em regime estável, o custo por readback em cada modo é:

- **Shadow:** a cópia GPU. Não há memcpy nem syscall: a proteção sobrevive à supersessão do mesmo
  range. Isso é mais barato que o legado, que faz memcpy e arma/desarma watch a cada scope.
- **Direct:** nada, se ninguém consome antes da próxima escrita. Se alguém consome, uma única cópia
  image→buffer em vez de duas (image→shadow e depois shadow→buffer).
- **CpuCommit:** idêntico ao legado.

### 4.2 Classificação por consumidor (aprendida, sem título)

- **Chave:** `(guest_begin, download_size)`. RTs e pools costumam repetir endereços entre frames.
  Tabela pequena, com endereçamento aberto e decaimento.
- **Padrão:** GPU-only, portanto Shadow (e Direct depois da F4).
- **Transições:**
  - uma leitura da CPU que toca **bytes da authority** marca `CpuConsumer`. A partir dali, as
    próximas versões do range usam CpuCommit;
  - faults de false sharing acima de um limiar por página (§4.5) marcam `HotPage`. Os ranges
    daquela página passam a usar CpuCommit;
  - N frames sem leitura na CPU → volta a GPU-only.
- **No primeiro contato:** um range que a CPU lê paga uma única vez uma espera equivalente à do
  legado, dentro do fault. Depois disso fica no comportamento legado exato. Não há regressão
  sustentada.

### 4.3 Fila ordenada de publicação

Uma fila única, em ordem de programa, no tracker:

- `EagerWriteback { seq, tick }`: todo readback CpuCommit ou Legacy. Termina quando o commit do
  priority thread roda (`DownloadImageMemory`, com `SCOPE_EXIT` no lambda).
- `Signal { seq, tick, label, value (64 bits), bytes, publish(bool write_label) }`: todo sinal
  adiado (EOS SignalFence, EOP Data32/Data64 com ou sem IRQ, ReleaseMem).

Regras:

1. Um scope **sem** downloads publica no tempo do CP, como hoje. Nada muda.
2. Um scope com downloads publica no tempo do CP se **não houver nenhum `EagerWriteback` pendente**.
   Nesse caso só authorities estão em voo.
3. Caso contrário, o sinal entra na fila e é publicado na conclusão (`PublishThrough`), depois dos
   writebacks anteriores. É o invariante do legado.
4. **Wait no CP (GFX e ASC):**
   - se a RAM já satisfaz a condição, passa (como hoje);
   - se um sinal pendente satisfaz a condição (`TestWaitValue` sobre o valor pendente; qualquer
     função, máscara, ref, valor, 32 ou 64 bits) e todos os `EagerWriteback` anteriores a ele já
     terminaram, publica agora o prefixo da fila até ele e passa. No hardware, o CP só passa desse
     wait quando o label já está na memória, então a ordem é exata. CP-writes posteriores ao label
     (reset) ficam naturalmente depois dele;
   - senão, legado: flush + spin.
5. Cada sinal é publicado exatamente uma vez e em ordem. Dois mutexes:
   - `publish_mutex` serializa a execução das publicações;
   - `timeline_mutex` protege a fila; nenhuma publicação roda segurando ele.

   O CP usa try-lock. Se o priority thread está publicando, o CP cai no caminho legado, que
   termina logo. Assim não há deadlock com locks de cache.
6. **Unmap:** um sinal cujo label está numa faixa que vai morrer não escreve o label, mas mantém o
   IRQ. Isso também corrige um risco que já existe hoje nos sinais adiados do legado.

Com as regras 1–4, a ordem que a CPU observa é a do programa. Não existe mais "wait pulado com
label escrito depois", então os riscos de ordem da versão atual desaparecem por construção:

- reset do label por `WRITE_DATA` sobrescrito pela escrita atrasada;
- EOP imediato ultrapassando o label anterior.

### 4.4 Correções necessárias no tracker

| Hoje | Correção |
|---|---|
| sobreposição parcial: uma authority antiga materializada depois da nova sobrescreve bytes novos (nunca aconteceu com 512 B fixos) | materializar sempre as authorities mais antigas sobrepostas antes da mais nova |
| materialização numa thread do guest com o tick produtor ainda não submetido espera sem ninguém submeter | `SendCommand` para o CP fazer o flush (o mesmo padrão de `BufferCache::ReadMemory`) |
| `TryWriteBacking` (commit legado, download de buffer) sobre um range de authority não avisa o tracker; a authority depois materializa bytes velhos por cima | escrita pela backing supersede ou recorta a authority. Exceção: a própria materialização |
| `HandleCpuWrite` lê o endereço para telemetria dentro do fault handler e agora rodaria em todo jogo | só com `HeavyEnabled` |
| todas as funções do tracker chamam `IsGow3FastpathActive()`, que compara uma string a cada chamada | contador atômico de authorities vivas; sem authorities, retorno imediato sem lock |
| `CollectGpuShadowPieces` sem nenhuma peça "assume" qualquer faixa protegida | só assume faixas que tocam páginas de authority |
| reuso direto do shadow só para exatamente 512 B | qualquer subfaixa contida num único shadow, com offset alinhado a 16 bytes |
| wait virtual só com ref 1 e máscara cheia; EOS só com valor 1; apenas um candidato; exatamente um download pendente | removido: substituído pela fila (§4.3) e pelo drain por candidato |

### 4.5 Page faults sem espera

**a) Código do emulador.** Aqui a resposta à sua pergunta é sim: o mecanismo do `e03d11c` se
aplica, e deve virar regra. Hoje só o resolver do copy engine lê pela backing view os bytes que não
são da authority (`buffer_cache.cpp:1846-1875`, `guest_copy_engine.cpp:607`). Outros leitores ainda
leem pelo VA e caem em fault (materialização, possível espera):

- hash de textura: `XXH3_64bits` em `texture_cache.cpp:1484`, `2487` e `2517`;
- `CopySparseMemory` nos caminhos não adiados (`buffer_cache.cpp:1825`, `2344`, `2388`);
- `StreamBuffer::Copy`;
- leituras do CP em memória guest.

Todos passam a usar uma função única com três casos:

- bytes de authority → shadow ou imagem na GPU;
- demais bytes de página protegida → `ReadBacking`;
- resto → leitura normal.

`MemoryManager::ReadBacking` e `IsBackedRange` já existem (`memory.cpp:690-700`).

**b) Código do guest.** O guest não pode ser redirecionado para a backing sem emular a instrução,
por isso o trick dos workers sozinho não resolve. O handler de fault decodifica a instrução com o
`Common::Decoder` (Zydis, que já existe e é usado em `signals.cpp`) para obter o intervalo exato e
se é leitura ou escrita:

| Acesso | Ação | Espera |
|---|---|---|
| não toca bytes de authority (false sharing) | emular load/store via backing: GPR e XMM primeiro; nas outras instruções, single-step (infra de `RequestSingleStepRearm`). A página continua protegida. Conta para `HotPage` | nenhuma |
| toca authority e a GPU já terminou | memcpy do shadow para a RAM, desprotege, marca `CpuConsumer` | nenhuma |
| toca authority e a GPU não terminou | espera só aquele tick (com flush via `SendCommand` se necessário), materializa, marca `CpuConsumer` | a mesma que o legado teria no label, e só uma vez por range |

### 4.6 Direct authority (F4)

- No scope não existe cópia, `EndRendering` nem transição. A authority é
  `(image uid, epoch, faixa)`.
- **Consumidor de buffer:** `SynchronizeBufferFromImage`, que já existe e já copia da imagem.
- **Consumidor de imagem:** a própria imagem.
- **Copy-on-overwrite:** antes de gravar a próxima escrita da imagem (`MarkWrite`, escrita de
  alias, destruição, GC, unmap), se a authority ainda estiver viva e sem consumidor, grava a cópia
  para o shadow nesse momento. No pior caso o custo é igual ao da F1, nunca maior.
- **Leitura da CPU:** o CP grava a cópia e faz flush; a thread espera só aquele tick (fault
  handler, via `SendCommand`).
- **Retirada** (48 ticks ou pressão): shadow → RAM.

---

## 5. Barriers, render pass e submits (vale para todos os jogos)

### 5.1 Epoch de flush no lugar de `FlushCaches`/`AcquireMemory` globais

- **Pacotes de cache do guest** (`EVENT_WRITE` de flush, `ACQUIRE_MEM`, EOS/EOP/ReleaseMem, wait do
  CP, submit) só incrementam `flush_epoch` e acumulam os estágios de origem. Não gravam nenhum
  comando Vulkan.
- **Buffers e imagens** guardam `write_epoch` da última escrita ainda não sincronizada.
- **No bind** (buffers, texturas, RT, indirect), um recurso com escrita pendente e
  `write_epoch < flush_epoch` recebe uma barrier dirigida, mesmo que a máscara de acesso não mude.
  Isso fecha o buraco RW→RW do `Buffer::GetBarrier` do jeito certo. A barrier entra na lista de
  barriers do draw que já existe: um único `pipelineBarrier2` e, no máximo, um `EndRendering`
  quando realmente houver dependência.
- **Sem flush do guest entre escrita e uso**, nenhuma barrier é gravada. No hardware isso é corrida;
  o guest que depende de ordem emite o flush.
- **Escritas não rastreáveis** (pipelines com `uses_dma`/BDA que escrevem) marcam uma flag. O
  próximo flush do guest vira a barrier global de hoje, apenas nesse caso.
- **Resultado:** `EndRendering` + barrier global por pacote se torna uma barrier estreita apenas
  nos recursos realmente escritos e depois lidos.
- **Mesma lógica para storage images RW→RW:** hoje `Image::GetBarriers` sempre emite barrier depois
  de uma escrita. Esta parte fica numa subfase separada, validada por sync validation.

### 5.2 Drain sem `EndRendering` incondicional

O `EndRendering` passa a acontecer apenas quando uma cópia é gravada de fato. Com Direct, isso não
acontece no scope.

### 5.3 Barriers estreitas e em lote

- **Shadow serve:** trocar as memory barriers globais por buffer barriers nas faixas do ring de
  download e do destino. Agrupar por draw.
- **`SynchronizeBuffer`:** usar a faixa copiada e o estágio de origem rastreado pelo `Buffer`, no
  lugar de `ALL_COMMANDS` no buffer inteiro. Um par pré/pós por lote de uploads do draw, e não um
  par por buffer.

### 5.4 Política de alimentação da GPU (kick)

Hoje a GPU recebe trabalho no meio do frame como efeito colateral dos flushes `WaitProgress` nos
waits. Quando as esperas somem, a GPU pode ficar ociosa enquanto o CP grava ~10 ms de frame.

A regra passa a ser explícita, independente de sync:

> submeter quando a GPU estiver ociosa ou prestes a ficar (`KnownGpuTick` alcançou o último
> submetido) e houver trabalho acumulado acima de um limiar (draws ou µs de CP).

Com isso o flush de progresso do wait virtual e o `WritebackReleaseMem` deixam de ser obrigatórios
e ficam só onde um observador no host precisa.

### 5.5 EOS `GdsStore` sem `Finish()`

Grava a cópia GDS → ring de download e publica o valor pela fila (§4.3). Um wait sobre o endereço
antes da conclusão cai no legado (o valor ainda não existe). Ninguém mais espera a GPU inteira.

### 5.6 Limpeza

Remover o mecanismo morto: `TrackDeferredGpuCompletion` não tem chamador, e com ele saem
`TryBypassGpuCompletionWait`, `pending_gpu_fence_words` e `GpuFenceWait` (barrier total). Remover
também `TryPromoteGoW3Eos`, o singleton de candidato, `IsGow3FastpathActive`,
`SHADPS4_GPU_SHADOW_SERVE` e `VirtualGpuFence`.

---

## 6. Fases

Cada fase:

- é um conjunto de commits compiláveis;
- compara build contra build, na mesma cena e com o mesmo driver;
- só é aceita se não houver regressão em nenhum jogo da matriz (§8).

### F0: baseline (sem mudança de comportamento)

- Adicionar contadores:
  - modo por readback;
  - sinais publicados no CP, no wait e na conclusão;
  - faults por classe (authority, false sharing emulado, false sharing por single-step);
  - classificações;
  - kicks.
- Capturar a baseline do `26dc6a7` em cada jogo da matriz, com build Release (FPS e frametime
  p50/p95/p99) e build de telemetria.

### F1: readback lazy generalizado

- Remover gates e filtros (§4.4, §5.6). Drain por candidato: authority Shadow quando elegível,
  legado caso contrário.
- Fila ordenada de publicação (§4.3).
- Classificação mínima: leitura da CPU em bytes da authority → CpuCommit nas próximas versões.
- Correções do tracker (§4.4).
- Kick (§5.4).
- **Não-regressão:**
  - jogos com readback desligado: nenhum caminho novo (sem pendências, portanto sem authorities);
  - consumidores na CPU: voltam ao legado depois do primeiro contato;
  - GoW3: o caso atual passa pela mesma regra, sem flush obrigatório no wait.
- **Esperado:**
  - `WaitRegMemSpinNs` e `HostWait(FenceCpuVisibility)` caem nos jogos com readback;
  - storage images de compute do GoW3 também saem do legado;
  - memcpy de readback some nos ranges só de GPU.

### F2: faults sem espera

- Leitores do emulador pela backing view (§4.5a).
- Decodificação e emulação no fault do guest (§4.5b).
- `HotPage`.
- **Esperado:** zero materializações com espera causadas por false sharing.
  `AuthorityMaterializeNs` perto de zero nos ranges só de GPU.

### F3: epoch de flush

- §5.1, primeiro para buffers; storage images RW→RW numa subfase.
- **Esperado:** `EventWriteFlushBarriers` e `AcquireMemBarriers` perto de zero; menos
  `EndRendering` por `RequiredMemoryDependency`; tempo GPU `DependencyDelay` menor.
- **Gate:** run curta com Vulkan Synchronization Validation sem novos hazards.

### F4: direct authority

- §4.6.
- **Esperado:** menos `EndRendering` e transições por readback; bytes de cópia menores.

### F5: estreitamento, lotes e restos

- §5.2, §5.3, §5.5 e a limpeza.
- **Esperado:** menos barriers por draw, menos submits por frame, sem `Finish` em GDS.

**Por que essa ordem:**

- F1 é a generalização direta do mecanismo que já provou funcionar e entrega o ganho do GoW3 aos
  outros jogos.
- F2 remove o maior risco de regressão do lazy (false sharing).
- F3 é provavelmente o maior ganho do lado da GPU, mas mexe em ordenação e exige validação própria.
- F4 e F5 refinam custos que F1–F3 deixam visíveis.

---

## 7. Invariantes de correção

1. Um sinal com downloads nunca é publicado antes do commit de um `EagerWriteback` anterior.
2. Um sinal é publicado exatamente uma vez, na ordem de programa, pelo CP (wait) ou pelo priority
   thread (conclusão).
3. Bytes de authority só chegam à RAM via materialização. Nenhuma escrita pela backing sobre eles
   passa despercebida.
4. Materialização respeita a ordem de registro nas sobreposições.
5. Espera no host só por bytes pedidos e ainda não produzidos, e sempre pelo tick específico, nunca
   `Finish`.
6. Nenhum fault é gerado pelo emulador em página de authority.
7. Barrier é emitida quando um recurso escrito é usado depois de um flush do guest. Escritas não
   rastreáveis mantêm a barrier global.
8. Nenhuma decisão usa título, formato, tamanho, valor de label ou assinatura de wait.

---

## 8. Validação (você compila e testa)

### Matriz de jogos

- **GoW3:** cena de referência mais o loading e a shock wave.
- **2–3 jogos que precisam de Readback Linear Images:** de preferência um que leia RT na CPU
  (luminância, screenshot).
- **2 jogos com readback desligado:** controle; nada pode mudar neles.

### Métricas por fase (telemetria já existente e as novas da F0)

- frametime p50/p95/p99 na build Release;
- `HostWait` por razão e `WaitRegMemSpinNs`;
- submits por razão;
- `ScopeBreak` por razão e evitabilidade;
- `BarrierCalls`, `EventWriteFlushBarriers`, `AcquireMemBarriers`, `BruteForceBarriers`;
- intervalos GPU `DependencyDelay`;
- `Authority*` (created, materializations, retirements, materialize ns);
- faults por classe;
- `GuestCopyGpuServed*`;
- `PriorityOpsWaitNs`.

### Checagens visuais

- exposição e luminância, pós-processo, UI;
- conteúdo que a CPU lê (se o jogo tiver);
- ausência de flicker e corrupção.

### Critério de aceite

Nenhum jogo da matriz piora em p50/p95. Os contadores da fase mudam na direção esperada.

---

## 9. Riscos residuais e como aparecem

| Risco | Detecção | Resposta |
|---|---|---|
| Mesma memória física mapeada em dois VAs: a proteção cobre só um | verificar na criação da authority se o físico tem outro mapeamento | se tiver, usar CpuCommit para aquele range |
| Single-step (instrução não emulada) abre uma janela de µs com a página RW | contador por instrução | emular as instruções que mais aparecem |
| Classificação atrasada: o primeiro frame de um range com consumidor na CPU espera no fault | contador de `CpuConsumer` novo | esperado: uma vez por range por sessão, igual ao legado |
| Pressão no ring de download por shadows pinados | `StreamBufferWaitNs` e reclaims | retirada por orçamento (já existe) e Direct na F4 |
| Epoch de flush perdendo um hazard real | sync validation e checagem visual | a subfase de storage images é separada; `uses_dma` mantém a barrier global |

---

## 10. Mapa por arquivo

- `src/video_core/gpu_authority_tracker.h/.cpp`: sem gates; contador atômico; fila de publicação;
  ordem de materialização; subfaixa de shadow; classificação; `HandleBackingWrite`; unmap de
  labels; remoção das fences virtuais.
- `src/video_core/texture_cache/texture_cache.h/.cpp`: candidatos sem filtro; drain por candidato
  com dedup por imagem; registro fora dos locks; `EagerWriteback` no `DownloadImageMemory`; hash
  pela backing; `WaitGpuAuthorityShadow` fora do CP via `SendCommand`; Direct (F4).
- `src/video_core/amdgpu/liverpool.h/.cpp`: sem `TryPromoteGoW3Eos`; sinais pela fila; wait GFX/ASC
  com publicação antecipada; epochs de flush (F3); GDS (F5); remoção do mecanismo morto.
- `src/video_core/buffer_cache/buffer_cache.cpp`, `buffer.h`: subfaixa de shadow;
  `write_epoch`/barriers dirigidas; barriers estreitas e em lote; leitura pela backing.
- `src/video_core/renderer_vulkan/vk_rasterizer.cpp`: sem `SHADPS4_GPU_SHADOW_SERVE`;
  `FlushCaches`/`AcquireMemory` viram epochs (F3); flag de escrita DMA; kick.
- `src/video_core/renderer_vulkan/vk_scheduler.*`: política de kick.
- `src/video_core/page_manager.cpp`, `src/core/signals.cpp`: decodificação e emulação no fault.
- `src/core/memory.cpp`: notificação de escrita pela backing ao tracker.
- `src/video_core/texture_cache/image.*`: `write_epoch`; gancho de copy-on-overwrite (F4).
- `src/common/performance_telemetry.*`: contadores da F0; remoção dos registros específicos do GoW3
  (`title_id_hash`).

---

## 11. Respostas diretas

- **Readback Linear Images:** o fast path exige a opção **ligada**. Sem ela não há candidato. Ela
  continua sendo uma escolha de precisão do usuário. Com ela desligada, este plano não muda nada.
- **Barriers e semaphores desnecessários custam, sim.**
  - Uma barrier drena estágios da GPU e costuma quebrar o render pass. O custo é tanto maior quanto
    mais trabalho estiver em voo e quanto mais largas forem as máscaras.
  - Aqui só existe uma fila e um timeline semaphore. O custo "de semaphore" aparece como submits
    extras e esperas no host em ticks.
  - Nada disso é perigoso para a correção, mas é desempenho perdido. §5 ataca cada caso.
- **O mecanismo de leitura sem fault dos workers se aplica ao readback.** Para o código do
  emulador, é o mesmo trick (backing view) estendido a todos os leitores (§4.5a). Para o código do
  guest, precisa de decodificação e emulação da instrução no fault (§4.5b). Com isso, nenhum acesso
  a bytes vizinhos espera a GPU.
- **Pendente:** as outras variáveis de ambiente (`SHADPS4_VK_RECORD_THREAD`,
  `SHADPS4_ASYNC_COPIES`, `SHADPS4_COPY_WORKERS`, `SHADPS4_VK_RECORD_AUDIT`,
  `SHADPS4_VERIFY_STAGE_SHAPE`) não são por jogo. Falta decidir se saem também.
