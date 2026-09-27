"""命令行入口。

用法示例：
    vedioai check                       # 检查凭证与依赖
    vedioai ingest "D:\\courses\\lesson1.mp4"
    vedioai ingest-docs --folder "D:\\notes"
    vedioai list
    vedioai ask <video_id> "老师说的三种排序分别是什么"
    vedioai notes <video_id>
    vedioai serve                       # 起本地 Web UI
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .ask import AskService
from .config import load_config
from .embedding import Embedder, Reranker, build_embedder, build_local_models
from .llm.client import from_llm_config
from .ingest.asr_volc import VolcASRClient
from .llm.client import LLMClient, LLMError
from .context import estimate_tokens, summary_without_outline
from . import ledger
from .notes import NotesService
from .pipeline import IngestPipeline, video_id_for
from .schema import ms_to_hms
from .store import Store

DEFAULT_QUESTIONS = "evals/questions.yaml"


def _setup_logging(verbose: bool) -> None:
    # Windows 控制台默认 GBK，中文日志会乱码；固定成 UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # 第三方库在默认级别下噪音太大，会把我们自己的进度信息冲掉：
    # - jieba 会把 logger 设成 DEBUG 并自行输出分词初始化过程；
    # - httpx/httpcore 按 INFO 记录每一次请求（含 TLS 握手细节）。
    # 需要排查网络或分词问题时加 -v 再打开。
    if not verbose:
        for name in ("jieba", "httpx", "httpcore", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _make_clients(cfg, *, need_asr=False, need_llm=False, need_vision=False, store=None):
    asr = None
    llm = None
    vision = None
    if need_asr:
        asr = VolcASRClient(cfg.asr)
    if need_llm:
        llm = ledger.attach(from_llm_config(cfg.llm), store)
    if need_vision and cfg.vision.api_key:
        vision = ledger.attach(
            LLMClient(
                cfg.vision.api_key,
                cfg.vision.base_url,
                cfg.vision.model,
                temperature=0.2,
                max_tokens=4096,
            ),
            store,
        )
    return asr, llm, vision


def _resolve_video_id(store: Store, token: str) -> str:
    """支持直接传视频路径（自动算 ID）或传 video_id / 标题片段。"""
    if store.get_video(token):
        return token
    path = Path(token)
    if path.exists():
        return video_id_for(path)
    matches = [v for v in store.list_videos() if token.lower() in (v["title"] or "").lower()]
    if len(matches) == 1:
        return matches[0]["video_id"]
    if len(matches) > 1:
        raise SystemExit(f"匹配到多个课程，请用 video_id 指定：{[m['video_id'] for m in matches]}")
    raise SystemExit(f"未找到课程：{token}")


# --------------------------------------------------------------------- 命令


def cmd_check(args, cfg) -> int:
    print("=" * 64)
    print("依赖与凭证检查")
    print("=" * 64)

    print(f"数据目录      : {cfg.data_dir}")
    print(f"数据库        : {cfg.db_path}")

    # FFmpeg
    from .ingest.media import probe  # noqa: F401

    try:
        import subprocess

        out = subprocess.run(
            [cfg.media.ffmpeg, "-version"], capture_output=True, text=True, timeout=10
        )
        first = (out.stdout or "").splitlines()[0] if out.stdout else "未知"
        print(f"FFmpeg        : OK  {first[:70]}")
    except Exception as exc:  # noqa: BLE001
        print(f"FFmpeg        : 缺失（{exc}）")

    # ASR
    print(f"火山 ASR      : {'OK' if cfg.asr.ready else '未配置'}  resource={cfg.asr.resource_id}")

    # LLM
    print(f"DeepSeek      : {'OK' if cfg.llm.api_key else '未配置'}  model={cfg.llm.model}")

    # Vision
    print(f"视觉模型      : {'OK' if cfg.vision.api_key else '未配置'}  model={cfg.vision.model}")

    # OCR
    try:
        import rapidocr_onnxruntime  # noqa: F401

        print("RapidOCR      : OK")
    except ImportError:
        print("RapidOCR      : 未安装（课件 OCR 将跳过）pip install rapidocr-onnxruntime")

    # yt-dlp / Cookie
    from .ingest.download import cookie_status, ytdlp_available

    print(f"yt-dlp        : {'OK' if ytdlp_available() else '未安装（URL 入库不可用）pip install yt-dlp'}")
    st = cookie_status(cfg.data_dir)
    if st["present"]:
        print(f"登录 Cookie   : OK  {st['lines']} 条（{st['path']}）")
    else:
        print("登录 Cookie   : 未导入（B 站大会员等需登录内容时用 vedioai cookies import）")

    # 本地模型
    embedder = build_embedder(cfg)
    if embedder.available:
        backend = "ONNX" if isinstance(embedder, Embedder) else "Ollama"
        print(f"本地嵌入      : OK  {backend} 可用（向量召回已启用）")
    else:
        reason = getattr(embedder, "reason", "") or "未找到模型"
        print(f"本地嵌入      : 未启用（{reason}）→ 检索退化为关键词")
    reranker = Reranker(cfg.rerank_model_dir)
    print(f"本地重排      : {'OK' if reranker.available else '未找到（不做重排）'}")

    print("=" * 64)
    if args.live:
        print("实时连通性检查（会发真实请求，消耗极少量额度）")
        print("-" * 64)
        from .selfcheck import run_live_checks

        results = run_live_checks(cfg)
        for r in results:
            print(f"{r.name:<12}: {r.mark}  {r.detail}")
        failed = [r.name for r in results if not r.ok]
        print("-" * 64)
        if failed:
            print(f"未通过：{'、'.join(failed)}")
            return 1
        print("全部可用。")
        return 0

    print("=" * 64)
    if not cfg.asr.ready:
        print("提示：复制 .env.example 为 .env 并填入火山语音的 Key。")
    if not cfg.llm.api_key:
        print("提示：在 .env 中填入 DEEPSEEK_API_KEY。")
    print("提示：加 --live 可以发真实请求验证密钥是否真的可用（推荐第一次配置后跑一次）。")
    return 0


def cmd_ingest(args, cfg) -> int:
    # 先校验文件，再要凭证：路径写错时给出的是「文件不存在」，
    # 而不是让人去查密钥，少一次无谓排查
    from .ingest.download import (
        cookie_status,
        cookies_file,
        download_video,
        looks_like_url,
        ytdlp_available,
    )

    source = args.video
    title_override = None
    video_id_override = None
    video_path: Path

    if looks_like_url(source):
        if not ytdlp_available():
            print("错误：未安装 yt-dlp。pip install yt-dlp", file=sys.stderr)
            return 1
        from .ingest.download import source_id_for_url

        vid = source_id_for_url(source)
        cookies = cookies_file(cfg.data_dir) if cookie_status(cfg.data_dir)["present"] else None
        print(f"拉取视频：{source}")
        if cookies:
            print(f"使用 Cookie：{cookies}")
        else:
            print("未配置 Cookie（公开内容通常够用；大会员/需登录时请先：vedioai cookies import <文件>）")

        def on_dl(pct, message):
            sys.stdout.write(f"\r  {message:<60}")
            sys.stdout.flush()

        result = download_video(
            source,
            cfg.downloads_dir / vid,
            cookies=cookies,
            ffmpeg=cfg.media.ffmpeg,
            progress=on_dl,
        )
        sys.stdout.write("\n")
        video_path = result.path
        title_override = result.title
        video_id_override = result.source_id
        print(f"已下载：{result.title}")
    else:
        video_path = Path(source)
        if not video_path.exists():
            print(f"错误：视频文件不存在：{video_path}", file=sys.stderr)
            return 1

    store = Store(cfg.db_path)
    asr, llm, _ = _make_clients(cfg, need_asr=True, need_llm=not args.no_summary, store=store)
    embedder = build_embedder(cfg)

    if args.no_slides:
        cfg.slides.enabled = False

    pipeline = IngestPipeline(cfg, store, llm=llm, embedder=embedder, asr=asr)

    def on_progress(p):
        bar_total = 30
        filled = int(p.percent / 100 * bar_total) if p.total else 0
        bar = "█" * filled + "·" * (bar_total - filled)
        sys.stdout.write(f"\r[{bar}] {p.percent:5.1f}%  {p.message[:48]:<48}")
        sys.stdout.flush()

    video = pipeline.run(
        video_path,
        skip_summary=args.no_summary,
        reuse_slides=not args.refresh_slides,
        progress=on_progress,
        title=title_override,
        video_id=video_id_override,
    )
    sys.stdout.write("\n")
    print(f"完成：{video.title}")
    print(f"  video_id : {video.video_id}")
    print(f"  时长     : {ms_to_hms(video.duration_ms)}")
    print(f"  产物目录 : {cfg.library_dir / video.video_id}")
    store.close()
    return 0


def cmd_ingest_docs(args, cfg) -> int:
    """批量导入 Markdown / 纯文本文档为课程（不生成视频笔记）。"""
    from .ingest.document import list_document_files

    paths = list_document_files(
        paths=[Path(p) for p in (args.paths or [])],
        folder=Path(args.folder) if args.folder else None,
    )
    if not paths:
        print("错误：未找到可导入的 .md / .txt / .markdown 文件", file=sys.stderr)
        return 1

    store = Store(cfg.db_path)
    _, llm, _ = _make_clients(cfg, need_asr=False, need_llm=not args.no_summary, store=store)
    embedder = build_embedder(cfg)
    pipeline = IngestPipeline(cfg, store, llm=llm, embedder=embedder, asr=None)

    ok = 0
    for i, doc_path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {doc_path}")

        def on_progress(p, _path=doc_path):
            bar_total = 30
            filled = int(p.percent / 100 * bar_total) if p.total else 0
            bar = "█" * filled + "·" * (bar_total - filled)
            sys.stdout.write(f"\r  [{bar}] {p.percent:5.1f}%  {p.message[:48]:<48}")
            sys.stdout.flush()

        try:
            video = pipeline.run_document(
                doc_path,
                skip_summary=args.no_summary,
                progress=on_progress,
            )
            sys.stdout.write("\n")
            print(f"  完成：{video.title} ({video.video_id})")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            sys.stdout.write("\n")
            print(f"  失败：{exc}", file=sys.stderr)

    store.close()
    print(f"合计 {ok}/{len(paths)} 门文档课入库成功")
    return 0 if ok == len(paths) else 1


def cmd_cookies(args, cfg) -> int:
    from .ingest.download import clear_cookies, cookie_status, save_cookies

    action = args.cookies_action
    if action == "status":
        st = cookie_status(cfg.data_dir)
        if st["present"]:
            print(f"已配置 Cookie：{st['path']}（{st['lines']} 条，{st['bytes']} 字节）")
        else:
            print("未配置 Cookie。")
            print("用浏览器扩展导出 Netscape cookies.txt 后：")
            print("  vedioai cookies import path\\to\\cookies.txt")
        return 0
    if action == "clear":
        clear_cookies(cfg.data_dir)
        print("已清除 Cookie。")
        return 0
    if action == "import":
        path = Path(args.file)
        if not path.exists():
            print(f"错误：文件不存在：{path}", file=sys.stderr)
            return 1
        try:
            st = save_cookies(cfg.data_dir, path.read_text(encoding="utf-8", errors="replace"))
        except ValueError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1
        print(f"已导入：{st['path']}（{st['lines']} 条）")
        return 0
    print(f"未知操作：{action}", file=sys.stderr)
    return 1


def cmd_list(args, cfg) -> int:
    store = Store(cfg.db_path)
    videos = store.list_videos()
    if not videos:
        print("课程库为空。用 `vedioai ingest <视频路径>` 入库。")
        return 0
    print(f"{'video_id':<18} {'类型':<6} {'状态':<10} {'时长':>8} {'章':>4} {'块':>5} {'图':>5}  标题")
    print("-" * 104)
    for v in videos:
        kind = "文档" if (v.get("kind") or "video") == "document" else "视频"
        print(
            f"{v['video_id']:<18} {kind:<6} {v['status']:<10} {ms_to_hms(v['duration_ms']):>8} "
            f"{v['chapter_count']:>4} {v['chunk_count']:>5} {v['slide_count']:>5}  {v['title']}"
        )
    store.close()
    return 0


def cmd_info(args, cfg) -> int:
    store = Store(cfg.db_path)
    video_id = _resolve_video_id(store, args.video)
    video = store.get_video(video_id)
    assert video is not None
    chapters = store.get_chapters(video_id)
    chunks = store.get_chunks(video_id)

    print(f"标题     : {video.title}")
    print(f"video_id : {video.video_id}")
    print(f"类型     : {video.kind.value}")
    print(f"状态     : {video.status.label}")
    print(f"时长     : {ms_to_hms(video.duration_ms)}")
    print(f"路径     : {video.path}")
    print(f"代理     : {video.proxy_path}")
    print(f"片段     : {len(chunks)} 块 / {len(chapters)} 章")
    summary = store.get_video_summary(video_id)
    if summary:
        print("\n全课摘要：")
        # 剥掉自带的「大纲：」列表：下面就会打印章节列表。
        # 顺带修掉一个显示问题：原来 summary[:1200] 会把 1200 字全花在
        # 那份列表上（摘要正文只有 352 字），正文反而被挤掉。
        print(summary_without_outline(summary)[:1200])

    if chapters:
        print("\n章节：")
        for ch in chapters:
            title = ch.title or f"第 {ch.idx + 1} 章"
            print(f"  {ch.idx + 1:>2}. [{ms_to_hms(ch.start_ms)}] {title}")
    store.close()
    return 0


def cmd_ask(args, cfg) -> int:
    store = Store(cfg.db_path)
    _, llm, vision = _make_clients(cfg, need_llm=True, need_vision=True, store=store)
    # 带上本地嵌入/重排模型（不存在时自动退化为关键词检索）
    embedder, reranker = build_local_models(cfg)
    service = AskService(cfg, store, llm, vision, embedder=embedder, reranker=reranker)
    video_id = _resolve_video_id(store, args.video)

    at_ms = None
    if args.at:
        at_ms = _parse_time(args.at)

    answer = service.ask(video_id, args.question, current_ms=at_ms, top_k=args.top_k)

    print("=" * 72)
    print(answer.text)
    print("=" * 72)
    if answer.citations:
        print("\n引用位置：")
        for c in answer.citations:
            print(f"  [{c.label}] {c.text[:60].replace(chr(10), ' ')}")
    u = answer.usage
    print(
        f"\n意图={answer.intent}  前缀≈{answer.prefix_tokens} tokens  "
        f"输入={u.prompt_tokens}（缓存命中 {u.cached_tokens}，{u.cache_hit_rate:.0%}）  "
        f"输出={u.completion_tokens}"
    )
    store.close()
    return 0


def cmd_notes(args, cfg) -> int:
    store = Store(cfg.db_path)
    _, llm, _ = _make_clients(cfg, need_llm=True, store=store)
    video_id = _resolve_video_id(store, args.video)
    service = NotesService(cfg, store, llm)

    def on_progress(done, total, message):
        sys.stdout.write(f"\r{message:<50}")
        sys.stdout.flush()

    result = service.generate(video_id, progress=on_progress, force=args.force)
    sys.stdout.write("\n")

    if args.out:
        out = Path(args.out)
        out.write_text(result.markdown, encoding="utf-8")
        print(f"已写入：{out}")
    else:
        print(f"已写入：{result.path}")
    print(f"字数：{len(result.markdown)}")
    store.close()
    return 0


def cmd_summarize(args, cfg) -> int:
    """只重跑分层摘要（补章节标题），不重新转写 / OCR。"""
    from .schema import VideoStatus
    from .summarize import build_summary_tree

    store = Store(cfg.db_path)
    video_id = _resolve_video_id(store, args.video)
    video = store.get_video(video_id)
    assert video is not None
    chunks = store.get_chunks(video_id)
    chapters = store.get_chapters(video_id)
    if not chunks or not chapters:
        print("没有分段数据，无法摘要", file=sys.stderr)
        store.close()
        return 1
    _, llm, _ = _make_clients(cfg, need_llm=True, store=store)
    store.set_status(video_id, VideoStatus.SUMMARY)

    def on_progress(done, total, message=""):
        pct = (done / total * 100.0) if total else 0.0
        sys.stdout.write(f"\r[{pct:5.1f}%] {message:<48}")
        sys.stdout.flush()

    print(f"补跑摘要：{video.title}（{len(chunks)} 块 / {len(chapters)} 章）")
    result = build_summary_tree(
        llm, store, video, chunks, chapters, progress=on_progress
    )
    sys.stdout.write("\n")
    if result.video_title:
        video.title = result.video_title[:80]
    video.error = result.describe() if result.degraded else None
    video.status = VideoStatus.READY
    store.upsert_video(video)

    embedder = build_embedder(cfg)
    if embedder.available:
        print("重建向量…")
        fresh = store.get_chunks(video_id)
        vecs = embedder.encode([c.combined_text for c in fresh])
        store.upsert_embeddings(
            video_id, zip([c.chunk_id for c in fresh], vecs, strict=False)
        )

    print(result.describe() if result.degraded else "摘要完成。")
    for ch in store.get_chapters(video_id)[:10]:
        print(f"  {ch.idx + 1}. {ch.title or '（无标题）'}")
    store.close()
    return 1 if result.degraded else 0


def cmd_serve(args, cfg) -> int:
    import uvicorn

    from .server import create_app

    app = create_app(cfg)
    print(f"打开 http://127.0.0.1:{args.port}  （Ctrl+C 退出）")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_reindex(args, cfg) -> int:
    """只重算本地向量索引，不重新转写。"""
    from .pipeline import reindex_videos

    store = Store(cfg.db_path)
    embedder, reranker = build_local_models(cfg)

    if not embedder.available:
        reason = getattr(embedder, "reason", "")
        print(
            "没有可用的本地嵌入模型，无法重建向量索引。\n"
            f"  检查过的 ONNX 目录：{cfg.embed_model_dir}\n"
            "    → 需要 model.onnx 与 tokenizer.json\n"
            f"  Ollama 回退：{cfg.retrieve.ollama_url}（模型 {cfg.retrieve.ollama_embed_model}）\n"
            + (f"    原因：{reason}\n" if reason else "")
            + "  在此之前检索只用关键词（功能可用，召回质量略低）。",
            file=sys.stderr,
        )
        store.close()
        return 1

    video_ids = [_resolve_video_id(store, args.video)] if args.video else None
    try:
        counts = reindex_videos(store, embedder, video_ids)
    except Exception as exc:  # noqa: BLE001
        print(f"重建索引失败：{exc}", file=sys.stderr)
        store.close()
        return 1

    if not counts:
        print("没有可重建的课程（课程库为空，或课程尚无文本块）。")
        store.close()
        return 0

    for video_id, n in counts.items():
        print(f"已重建 {video_id}：{n} 个文本块")
    print(f"完成：{len(counts)} 门课程、{sum(counts.values())} 个文本块已建立向量索引。")
    if not reranker.available:
        print("提示：未找到重排模型，检索将不做重排（可选，不影响可用性）。")
    store.close()
    return 0


def cmd_purge(args, cfg) -> int:
    store = Store(cfg.db_path)
    video_id = _resolve_video_id(store, args.video)
    IngestPipeline(cfg, store).purge(video_id)
    print(f"已删除 {video_id}")
    store.close()
    return 0


def cmd_repair(args, cfg) -> int:
    """按术语表纠正已入库文本（不重新转写，零 ASR 成本）。"""
    from .pipeline import repair_videos

    store = Store(cfg.db_path)
    video_ids = [_resolve_video_id(store, args.video)] if args.video else None

    embedder = None
    if not args.no_reembed:
        embedder, _ = build_local_models(cfg)
        if not embedder.available:
            print(
                "提示：没有可用的嵌入模型，本次只纠正文本、不重算向量。\n"
                "      文本改了但向量没改，会让检索召回变差；建议之后跑一次 reindex。",
                file=sys.stderr,
            )
            embedder = None

    if not cfg.glossary_path.exists():
        print(f"未找到术语表：{cfg.glossary_path}", file=sys.stderr)
        store.close()
        return 1

    try:
        report = repair_videos(cfg, store, video_ids, embedder=embedder, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"纠正失败：{exc}", file=sys.stderr)
        store.close()
        return 1

    field_label = {
        "segments": "转写句",
        "chunks": "文本块",
        "chapters": "章节",
        "slides": "课件 OCR",
        "video_summary": "全课摘要",
    }
    touched = 0
    for video_id, stats in report.items():
        total = sum(v for k, v in stats.items() if k != "reembedded")
        if not total:
            print(f"{video_id}：无需纠正")
            continue
        touched += 1
        detail = "、".join(
            f"{field_label.get(k, k)} {v} 处" for k, v in stats.items() if k != "reembedded"
        )
        tail = f"，并重算 {stats['reembedded']} 个向量" if "reembedded" in stats else ""
        print(f"{video_id}：{detail}{tail}")

    if args.dry_run:
        print(f"[试运行] 共 {touched} 门课程有可纠正内容，未写入。")
    else:
        print(f"完成：{touched} 门课程已纠正。")
    store.close()
    return 0


def cmd_reocr(args, cfg) -> int:
    """用原始分辨率重跑已入库课程的课件 OCR（不重新转写，零 ASR 成本）。

    什么时候需要：抽帧分辨率提高后，库里留着的仍是旧的低分辨率 OCR 结果——
    那些错字已经进了摘要和笔记，光改代码不会自动回补。
    """
    from .pipeline import reocr_video

    store = Store(cfg.db_path)
    try:
        video_id = _resolve_video_id(store, args.video)
    except SystemExit:
        store.close()
        raise

    embedder = None
    if not args.no_reembed:
        embedder, _ = build_local_models(cfg)
        if not embedder.available:
            print(
                "提示：没有可用的嵌入模型，本次只更新文本、不重算向量。",
                file=sys.stderr,
            )
            embedder = None

    ocr_width = cfg.slides.ocr_width or None
    where = "原始分辨率" if ocr_width is None else f"{ocr_width}px 宽"
    print(f"用{where}重跑课件 OCR（不重新转写）…")

    def on_progress(done: int, total: int, message: str) -> None:
        if total:
            print(f"  [{done}/{total}] {message}")

    try:
        stats = reocr_video(cfg, store, video_id, embedder=embedder, progress=on_progress)
    except Exception as exc:  # noqa: BLE001
        print(f"重跑 OCR 失败：{exc}", file=sys.stderr)
        store.close()
        return 1

    if not stats:
        print(f"{video_id}：库里没有课件，无需重跑。")
        store.close()
        return 0

    print(
        f"{video_id}：重跑 {stats.get('slides', 0)} 张课件，"
        f"更新 {stats.get('chunks', 0)} 个文本块"
        + (f"，并重算 {stats['reembedded']} 个向量" if "reembedded" in stats else "")
    )
    store.close()
    return 0


def cmd_usage(args, cfg) -> int:
    """查看用量账本；--backfill-asr 可回填已有课程的转写用量。"""
    store = Store(cfg.db_path)
    try:
        if args.backfill_asr:
            n = _backfill_asr_usage(store)
            print(f"已回填 {n} 门课程的转写用量。")

        data = store.usage_summary(args.video or None)
    finally:
        store.close()

    if not data["calls"]:
        print("账本为空。")
        print("提示：记账只覆盖开启本功能之后的调用；已有课程的转写可用")
        print("      `vedioai usage --backfill-asr` 回填（依据库里已有的转写时长）。")
        return 0

    def yuan(v: float) -> str:
        return f"¥{v:.4f}" if v < 1 else f"¥{v:.2f}"

    scope = f"课程 {args.video}" if args.video else "全库"
    print("=" * 64)
    print(f"{scope}用量：{data['calls']} 次调用，估算 {yuan(data['cost_yuan'])}")
    if data["unpriced_calls"]:
        print(
            f"注意：其中 {data['unpriced_calls']} 次调用没有价目，未计入金额"
            f"——实际花费高于上面这个数。"
        )
    print("=" * 64)
    print(
        f"输入 {data['prompt_tokens']:,} tokens"
        f"（命中缓存 {data['cached_tokens']:,}）"
        f"  输出 {data['completion_tokens']:,}"
    )
    if data["audio_ms"]:
        print(f"音频 {data['audio_ms'] / 60000:.1f} 分钟")

    print("\n按用途：")
    for r in data["by_kind"]:
        print(f"  {r['kind']:<10} {r['calls']:>4} 次   {yuan(r['cost'])}")

    print("\n按模型：")
    for r in data["by_model"]:
        print(f"  {r['model']:<26} {r['calls']:>4} 次   {yuan(r['cost'])}")

    if data["by_video"]:
        print("\n按课程：")
        for r in data["by_video"]:
            title = (r["title"] or r["video_id"])[:28]
            print(f"  {title:<30} {r['calls']:>4} 次   {yuan(r['cost'])}")

    if data["by_day"]:
        print("\n按天（近 30 天）：")
        for r in data["by_day"]:
            print(f"  {r['day']}  {r['calls']:>4} 次   {yuan(r['cost'])}")

    print(f"\n当前计费时段：{'高峰（价格翻倍）' if ledger.is_peak() else '闲时'}")
    print("金额为按官方价目估算，以控制台账单为准。")
    return 0


def _backfill_asr_usage(store: Store) -> int:
    """给已有课程补记转写用量。

    回填是**有据可依**的，不是编数字：库里存在转写句，说明转写确实跑过、
    确实按音频时长被计过费；而时长就在 videos 表里。所以这不是凭空生成，
    是把一个本来就发生过、只是当时没记的事实补上。

    已经有 asr 记录（或根本没有转写）的课程会跳过，重复执行不会翻倍。
    """
    done = 0
    for video in store.list_videos():
        vid = video["video_id"]
        if not video.get("duration_ms"):
            continue
        has_asr_row = any(
            r["kind"] == "asr" for r in store.usage_summary(vid)["by_kind"]
        )
        if has_asr_row:
            continue
        # 有转写句 = ASR 真跑过；没有就是还没走到那一步，不该记
        if not store.get_segments(vid):
            continue
        store.record_usage(
            kind="asr",
            model="volc.bigasr.auc_turbo",
            video_id=vid,
            audio_ms=video["duration_ms"],
            cost_yuan=ledger.estimate_asr_cost(video["duration_ms"]),
        )
        done += 1
    return done


def cmd_adjudicate(args, cfg) -> int:
    """让文本模型依据上下文判定术语表里待确认的词。"""
    from .glossary import (
        Glossary,
        adjudicate_suspects,
        _auto_path,
        render_verdicts,
        write_auto_glossary,
    )
    from .llm.client import from_llm_config

    store = Store(cfg.db_path)
    try:
        video_id = _resolve_video_id(store, args.video) if args.video else None
    except SystemExit:
        store.close()
        raise

    if video_id is None:
        videos = store.list_videos()
        if not videos:
            print("课程库为空。")
            store.close()
            return 1
        video_id = videos[0]["video_id"]

    glossary = Glossary.load(cfg.glossary_path, video_id)
    suspects = args.term or list(glossary.suspects)
    # 去重保序
    seen: set[str] = set()
    suspects = [s for s in suspects if not (s in seen or seen.add(s))]

    if not suspects:
        print("术语表里没有待确认的词（suspects 为空）。")
        store.close()
        return 0

    print(f"待判定 {len(suspects)} 个词，逐个收集上下文并交文本模型判断…")
    print(f"（载入的术语表：{', '.join(str(p.name) for p in glossary.sources) or '无'}）")

    try:
        client = from_llm_config(cfg.llm)
    except Exception as exc:  # noqa: BLE001
        print(f"无法构造文本模型客户端：{exc}", file=sys.stderr)
        store.close()
        return 1

    try:
        verdicts = adjudicate_suspects(glossary, store, client, video_id, terms=suspects)
    finally:
        client.close()
        store.close()

    if not verdicts:
        print("这些词在材料里找不到上下文，无法判定。")
        return 0

    print()
    print(render_verdicts(verdicts))
    print()

    applicable = [v for v in verdicts if v.applicable]
    uncertain = [v for v in verdicts if not v.applicable]
    print(f"可自动采纳：{len(applicable)} 条；仍需人工确认：{len(uncertain)} 条")
    for v in uncertain:
        print(f"  · {v.term} → {v.correct or '（未定）'}（{v.confidence}）{v.reason[:60]}")

    if args.apply and applicable:
        out = _auto_path(cfg.glossary_path)
        path, n = write_auto_glossary(out, verdicts)
        print()
        print(f"已写入 {n} 条到 {path.name}")
        print("下一步：vedioai repair   # 让纠正对已入库课程生效")
    elif applicable and not args.apply:
        print()
        print("加 --apply 可把上述高置信度结论写入 vedioai.glossary.auto.yaml。")
    return 0


def cmd_eval(args, cfg) -> int:
    from .eval_runner import main as run_eval
    from .eval_runner import scaffold

    if args.scaffold:
        # 注意：不要用字符串比较拼路径，Windows 上 Path 会渲染成反斜杠
        if args.questions == DEFAULT_QUESTIONS:
            out = Path("evals") / f"questions.{args.scaffold}.yaml"
        else:
            out = Path(args.questions)
        return scaffold(args.scaffold, cfg, out)

    return run_eval(args, cfg)


def _parse_time(text: str) -> int:
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return ((h * 60 + m) * 60 + s) * 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vedioai", description="课程视频 AI 理解")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    parser.add_argument("--config", type=Path, default=None, help="指定配置文件")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="检查依赖与凭证")
    p.add_argument("--live", action="store_true", help="发真实请求验证密钥是否可用")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("ingest", help="入库一门课程视频（本地路径或 URL）")
    p.add_argument("video", help="视频文件路径，或 B 站/YouTube/直链 URL")
    p.add_argument("--no-summary", action="store_true", help="跳过 LLM 摘要（只做转写与索引）")
    p.add_argument("--no-slides", action="store_true", help="跳过课件抽帧与 OCR")
    p.add_argument(
        "--refresh-slides",
        action="store_true",
        help="强制重新抽帧与 OCR（改了抽帧参数时用；默认会复用已有课件）",
    )
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ingest-docs", help="批量导入文档课程（Markdown / 纯文本）")
    p.add_argument("paths", nargs="*", help="一个或多个 .md / .txt / .markdown 路径")
    p.add_argument("--folder", type=Path, default=None, help="递归扫描该目录下的文档")
    p.add_argument("--no-summary", action="store_true", help="跳过 LLM 摘要（只做切分与索引）")
    p.set_defaults(func=cmd_ingest_docs)

    p = sub.add_parser("cookies", help="管理登录 Cookie（B 站等需登录站点）")
    csub = p.add_subparsers(dest="cookies_action", required=True)
    c1 = csub.add_parser("status", help="查看是否已导入 Cookie")
    c1.set_defaults(func=cmd_cookies)
    c2 = csub.add_parser("import", help="导入 Netscape cookies.txt")
    c2.add_argument("file", help="浏览器扩展导出的 cookies.txt 路径")
    c2.set_defaults(func=cmd_cookies)
    c3 = csub.add_parser("clear", help="清除已保存的 Cookie")
    c3.set_defaults(func=cmd_cookies)

    p = sub.add_parser("list", help="列出课程库")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="查看课程详情")
    p.add_argument("video")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("ask", help="就课程提问")
    p.add_argument("video")
    p.add_argument("question")
    p.add_argument("--at", help="当前播放位置，如 12:30")
    p.add_argument("--top-k", type=int, default=None)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("notes", help="生成学习文档")
    p.add_argument("video")
    p.add_argument("--out", help="输出 Markdown 路径")
    p.add_argument(
        "--force",
        action="store_true",
        help="即使多数章节生成失败也覆盖已有 notes.md（默认保留旧文档，避免用残次品盖掉良品）",
    )
    p.set_defaults(func=cmd_notes)

    p = sub.add_parser("summarize", help="只重跑分层摘要（补章节标题，不重新转写）")
    p.add_argument("video", help="课程 ID 或标题片段")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("serve", help="启动本地 Web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=17831)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("reindex", help="只重建本地向量索引（不重新转写，零 API 成本）")
    p.add_argument("video", nargs="?", help="课程 ID 或标题片段；省略则处理全部课程")
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("purge", help="删除某课程的入库产物")
    p.add_argument("video")
    p.set_defaults(func=cmd_purge)

    p = sub.add_parser("usage", help="查看用量账本（花了多少钱）")
    p.add_argument("video", nargs="?", help="课程 ID；省略则看全库")
    p.add_argument(
        "--backfill-asr",
        action="store_true",
        help="给开启记账之前入库的课程补记转写用量（依据库里已有的转写时长，不会重复记）",
    )
    p.set_defaults(func=cmd_usage)

    p = sub.add_parser(
        "repair",
        help="按术语表纠正已入库文本（不重新转写，零 ASR 成本；纠正 ASR 听错的名词）",
    )
    p.add_argument("video", nargs="?", help="课程 ID 或标题片段；省略则处理全部课程")
    p.add_argument("--dry-run", action="store_true", help="只统计将纠正多少处，不写入")
    p.add_argument(
        "--no-reembed",
        action="store_true",
        help="文本改后不重算向量（默认会重算；不重算会让检索召回变差）",
    )
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser(
        "reocr",
        help="用原始分辨率重跑已入库课程的课件 OCR（提高小字识别率，不重新转写）",
    )
    p.add_argument("video", help="课程 ID、标题片段或视频路径")
    p.add_argument("--no-reembed", action="store_true", help="不重算向量（默认会重算）")
    p.set_defaults(func=cmd_reocr)

    p = sub.add_parser(
        "adjudicate",
        help="让文本模型依据上下文判定术语表里待确认的词（如 ASR 听错的名词）",
    )
    p.add_argument("video", nargs="?", help="课程 ID 或标题片段；省略则取课程库第一门")
    p.add_argument("--term", action="append", help="只判定指定词，可重复；省略则用术语表的 suspects")
    p.add_argument("--apply", action="store_true", help="把可采纳的结论写入 vedioai.glossary.auto.yaml")
    p.set_defaults(func=cmd_adjudicate)

    p = sub.add_parser("eval", help="跑评估集（唯一裁判）")
    p.add_argument("questions", nargs="?", default=DEFAULT_QUESTIONS)
    p.add_argument("--out", default="evals/runs", help="报告输出目录")
    p.add_argument("--only", choices=["factual", "cross", "visual", "global"], help="只跑某一类题")
    p.add_argument("--scaffold", metavar="VIDEO_ID", help="按课程章节生成题目骨架")
    p.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)
    try:
        return args.func(args, cfg)
    except (LLMError, RuntimeError, FileNotFoundError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
