#!/usr/bin/env python3
"""Batch generate 5-10 second Japanese self-introduction audio samples for all cloned voice styles.

Each voice speaks in an expressive, deliberate manner tailored to their character persona,
and output is saved to voices/cloned_styles_japanese_samples/
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

# Add workspace to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helper import load_text_to_speech, load_voice_style, timer

MODEL_DIR = os.environ.get("MODEL_DIR", "supertonic3")
ONNX_DIR = os.path.join(MODEL_DIR, "onnx")
STYLES_DIR = Path("voices/cloned_styles")
OUTPUT_DIR = Path("voices/cloned_styles_japanese_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Character persona definitions: (Japanese Intro Text, Initial Speed)
CHARACTER_PROMPTS = {
    "012_Will_-_Relaxed_Optimist": (
        "はじめまして、ウィルです。焦らず深呼吸して、今日も気楽にいきましょうね。",
        0.86,
    ),
    "013_Jessica_-_Playful_Bright_Warm": (
        "やっほー、ジェシカだよ！今日もみんなにたくさんの笑顔と元気を届けにきたよ！",
        0.88,
    ),
    "014_Eric_-_Smooth_Trustworthy": (
        "はじめまして、エリックです。あなたの力になれるよう、誠心誠意サポートいたします。",
        0.86,
    ),
    "015_Bella_-_Professional_Bright_Warm": (
        "こんにちは、ベラです。明るく丁寧に、真心を込めてご案内させていただきますね。",
        0.88,
    ),
    "016_Chris_-_Charming_Down-to-Earth": (
        "どうも、クリスです！気取らず自然体で、楽しい時間を一緒に過ごしましょう。",
        0.87,
    ),
    "017_Brian_-_Deep_Resonant_and_Comforting": (
        "こんばんは、ブライアンです。静かな夜に、心安らぐ温かいひとときをお届けします。",
        0.84,
    ),
    "018_Daniel_-_Steady_Broadcaster": (
        "皆様、こんにちは。キャスターのダニエルです。本日の最新ニュースを正確にお伝えします。",
        0.88,
    ),
    "019_Lily_-_Velvety_Actress": (
        "ごきげんよう、リリーです。舞台の上から、美しい物語の世界へと皆様をお連れします。",
        0.85,
    ),
    "020_Adam_-_Dominant_Firm": (
        "アダムだ。迷わずついてこい。目標達成に向けて、確実にリードしてやる。",
        0.86,
    ),
    "021_Bill_-_Wise_Mature_Balanced": (
        "やあ、ビルだよ。長い人生、焦りは禁物だ。落ち着いて周りを見渡してみよう。",
        0.85,
    ),
    "022_Aakash_Aryan_-_Calm_and_Professional": (
        "アーカシュと申します。冷静かつ的確に、最適なソリューションをご提案します。",
        0.87,
    ),
    "023_Adam_Stone_-_Smooth_Relaxed_and_Deep": (
        "アダム・ストーンです。ゆったりと流れる時間の中で、心地よい響きをお届けします。",
        0.85,
    ),
    "024_Asahi_-_Calm_and_Natural": (
        "こんにちは、アサヒです。肩の力を抜いて、自然体でゆっくりお話ししましょう。",
        0.86,
    ),
    "025_Brian_-_Relatable_Everyman": (
        "どうも、ブライアンです！飾らない普段通りの会話を、一緒に楽しめたら嬉しいな。",
        0.87,
    ),
    "026_Bunty_-_Energetic_Expressive_Fun_Social_Media_Voice": (
        "みんな元気ー？ブンティーだよ！今日も最高にワクワクする情報をお届けしちゃうよ！",
        0.89,
    ),
    "027_Avani_-_Sensitive_Insurance_Claims_Agent": (
        "アヴァニと申します。ご不安な気持ちに寄り添い、丁寧にお手続きを支えいたします。",
        0.85,
    ),
    "028_Brenna": (
        "はじめまして、ブレナです。あなたの心にそっと寄り添えるような声をお届けしますね。",
        0.86,
    ),
    "029_Georg_-_Funny_and_Emotional": (
        "ハハッ、ゲオルグだよ！泣いて笑って、人生思いっきり楽しんでいこうじゃないか！",
        0.88,
    ),
    "030_Choy_Advertisting_Radio_TV_Online_Ads": (
        "注目！チョイです！今だけの特別なチャンスを、見逃さないでチェックしてください！",
        0.89,
    ),
    "031_Akari_-_Bright_Natural_Japanese_Female_-_Friendly_Voice": (
        "はじめまして、あかりです！明るく元気な声で、今日も素敵な一日を応援しますね！",
        0.88,
    ),
    "032_Bunty_Friendly_Fun_Customer_Agent": (
        "こんにちは！カスタマーサポートのブンティーです！困ったことがあれば何でも聞いてね！",
        0.89,
    ),
    "033_Cory_A_Radio_Voice_for_Technology_Slightly_Southern_Deep": (
        "テクノロジーの世界へようこそ、コーリーです。最先端の話題を分かりやすく語ります。",
        0.86,
    ),
    "034_Chris_Brift": (
        "こんにちは、クリスです。明瞭でスマートなコミュニケーションをお届けします。",
        0.87,
    ),
    "035_Aki_-_Friendly_Clear_and_Natural": (
        "はじめまして、アキです！すっきりと聞き取りやすい声で、心地よくお伝えします。",
        0.87,
    ),
    "036_Hope-_Your_conversational_bestie": (
        "ねえねえ、ホープだよ！何でも話してね、いつもあなたのそばにいる大親友だから！",
        0.88,
    ),
    "037_Harune_-_Japanese_Professional_Narration": (
        "皆様こんにちは、春音と申します。凛とした美しいナレーションで、物語を紡ぎます。",
        0.86,
    ),
    "038_Hiro_-_Calm_Measured_and_Husky": (
        "……ヒロだ。静寂の中に響く低いハスキーボイスで、落ち着いた時間をお届けしよう。",
        0.84,
    ),
    "039_Jessica_-_Playful_Bright_Warm": (
        "ジェシカだよ！ぽかぽか温かい日差しのように、みんなの心を明るく照らしちゃうよ！",
        0.88,
    ),
    "040_Jun_-_Calm_Clear_and_Husky": (
        "ジュンです。静けさの中に響くハスキーな声で、ゆっくりと言葉を紡いでいきます。",
        0.85,
    ),
    "041_Gojo_-_Calm_Clear_and_Measured": (
        "五条です。冷静に、そして正確に。物事の本質をしっかりと見極めていきましょう。",
        0.86,
    ),
    "042_Kmy_-_Characters": (
        "ふふっ、ボクはクミィ！不思議な世界の冒険へ、君を案内してあげるね！",
        0.87,
    ),
    "043_Christopher_Smooth": (
        "クリストファーと申します。滑らかで上質な時間を、私の声とともにお楽しみください。",
        0.85,
    ),
    "044_Kou_Sudo_-_Calm_Neutral_male": (
        "須藤航です。落ち着いたフラットな語り口で、自然な情報伝達を心がけています。",
        0.86,
    ),
    "045_Koichi_-_Japanese_Deep_Calm_Narrator": (
        "コウイチです。深みのある落ち着いた声で、重厚なナレーションをお届けします。",
        0.84,
    ),
    "046_Kazuha_-_Warm_Friendly": (
        "こんにちは、カズハです！温かいお茶を飲むような、ほっとする時間を過ごそうね。",
        0.86,
    ),
    "047_Kylie_-_Advertising_and_Social_Media": (
        "カイリーです！トレンドをいち早くキャッチして、最高に魅力的な発信を届けるよ！",
        0.88,
    ),
    "048_Ikuya_-_Authentic_Japanese_Salesperson": (
        "いらっしゃいませ、イクヤです！お客様にぴったりの素晴らしい商品をご提案します！",
        0.89,
    ),
    "049_Mira_-_Enerji_Kurumsal_Reklam": (
        "ミラです。先進的なビジョンと確かなエネルギーで、未来への扉を開きましょう！",
        0.87,
    ),
    "050_Ken_-_Friendly_Japanese_male": (
        "やあ、ケンです！親しみやすさをモットーに、今日も元気に話しかけていくよ！",
        0.87,
    ),
    "051_Alexandra_-_Conversational_and_Natural": (
        "はじめまして、アレクサンドラです。まるで隣でお茶を飲んでいるように話しましょう。",
        0.86,
    ),
    "052_Kenzo_-_Calm_Soft_and_Measured": (
        "ケンゾウと申します。穏やかで柔らかなトーンで、心地よい安らぎをお届けします。",
        0.85,
    ),
    "053_Morioki_-_Calm_Measured_and_Muffled": (
        "……モリオキです。静かな森の奥深くのように、静寂に満ちた声でお話しします。",
        0.84,
    ),
    "054_Preist": (
        "神の祝福がありますように。神父です。皆様の心に静けさと平安が宿るよう祈ります。",
        0.85,
    ),
    "055_Rie_-_Natural_Customer_Care_Agent": (
        "お電話ありがとうございます。リエです。お客様のお困りごとを優しくサポートします。",
        0.86,
    ),
    "056_Rie_-_Authentic_Outbound_Salesperson": (
        "こんにちは、リエと申します！本日は生活がもっと便利になる素敵なご案内がございます！",
        0.88,
    ),
    "057_Nguyen_-_Clear_Bright_Saleswoman": (
        "グエンです！明るくクリアな声で、とっておきのおすすめ情報を元気にお届けします！",
        0.88,
    ),
    "058_Ryo_-_A_calm_slightly_low_voice": (
        "リョウです。低めの落ち着いた声で、あなたの夜のひとときに寄り添いたいと思います。",
        0.85,
    ),
    "059_Koichi_Yashiro_Deep_Cinematic_Voice": (
        "八代浩一だ。映画のような壮大な物語を、この重厚な低音で描き出してみせよう。",
        0.84,
    ),
    "060_Mei_-_Friendly_Clear_and_Soft": (
        "メイだよ！やわらかくて澄んだ声で、みんなの疲れた心をふんわり癒やしてあげるね。",
        0.87,
    ),
    "061_Katelyn_-_Clear_Concise": (
        "ケイトリンです。要点を明瞭に、無駄なく的確にお伝えすることを大切にしています。",
        0.87,
    ),
    "062_Sawaro_-_Japanese_Actor_Voice": (
        "サワロだ。役者として培った情熱と表現力で、言葉の魂をあなたにお届けしよう。",
        0.85,
    ),
    "063_Rose_-_Inviting_Clear_and_Steady": (
        "ローズです。温かく迎え入れるような響きで、確かな安心感をお届けいたしますね。",
        0.86,
    ),
    "064_Titan_-_ConvoAI_ContentAI": (
        "システム起動。タイタンです。高度な対話処理と情報生成を迅速に実行いたします。",
        0.88,
    ),
    "065_Shohei_-_Warm_Clear_and_Husky": (
        "ショウヘイです。温もりを感じるハスキーな声で、親しみやすさを大切に語りかけます。",
        0.85,
    ),
    "066_Viraj_-_Warm_Energetic_and_Lively": (
        "ハロー！ヴィラージだよ！溢れるエネルギーと笑顔で、最高の一日をスタートしよう！",
        0.88,
    ),
    "067_Yoshiki_-_Everyday_Japanese_Narrator": (
        "ヨシキです。日常の風景を切り取るように、素朴で親しみやすい語りをお届けします。",
        0.86,
    ),
    "068_Suisen_-_Native_Japanese_female": (
        "水仙と申します。日本の四季折々の美しさを、たおやかな声にのせてお届けいたします。",
        0.85,
    ),
    "069_Michael_-_Genuine_and_Approachable": (
        "マイケルです！気取らない素直な言葉で、いつでも気楽にお話しできたら嬉しいです。",
        0.87,
    ),
    "070_Yuko_-_Calm_Clear_and_Steady": (
        "ユウコです。澄み渡る落ち着いた声で、大切なメッセージを丁寧にお届けいたします。",
        0.86,
    ),
    "071_Yoshi_M_-_Calm_Clear_Customer_Care": (
        "ヨシです。お客様の安心と満足を第一に、誠実で分かりやすい応対を心がけております。",
        0.87,
    ),
    "072_Yukari": (
        "ゆかりです。包み込むような温かな声で、あなたの毎日に優しい彩りを添えられますように。",
        0.86,
    ),
    "073_Yuki": (
        "ユキです。冬の澄んだ空気のように、清らかで静かな語りをお届けいたします。",
        0.85,
    ),
    "074_Zach_-_Storyteller_Narrator_Audiobooks_Podcasts": (
        "ザックです。さあ、本を開きましょう。心躍る物語の世界へ、あなたをお連れします。",
        0.85,
    ),
    "075_Yoko_Honda_-_Soft_female_voice": (
        "本田陽子です。やわらかく包み込むような優しい声で、心地よい安らぎをお届けします。",
        0.86,
    ),
    "076_Stacy_-_Sweet_and_Cute_English_Mandarin_female_voice": (
        "ステイシーだよ！甘くて可愛い声で、みんなの毎日をもっとハッピーにしちゃうよ！",
        0.88,
    ),
}


def synthesize_with_duration_guard(tts, text: str, voice_style, target_speed: float, min_sec: float = 5.0, max_sec: float = 10.0):
    """Synthesizes speech and guarantees duration is between min_sec and max_sec."""
    current_speed = target_speed
    max_attempts = 4
    
    for attempt in range(max_attempts):
        wav, dur = tts(text, "ja", voice_style, total_step=6, speed=current_speed)
        dur_sec = dur[0].item()
        
        if dur_sec < min_sec:
            # Too short: try slower speed if possible
            if current_speed > 0.65:
                current_speed = max(0.65, current_speed * 0.85)
                continue
            else:
                # If already very slow, pad silence at beginning and end
                trimmed = wav[0, : int(tts.sample_rate * dur_sec)]
                needed_padding = int((min_sec - dur_sec + 0.1) * tts.sample_rate)
                pad_front = np.zeros((needed_padding // 2,), dtype=np.float32)
                pad_back = np.zeros((needed_padding - needed_padding // 2,), dtype=np.float32)
                trimmed = np.concatenate([pad_front, trimmed, pad_back])
                final_dur = len(trimmed) / tts.sample_rate
                return trimmed, final_dur
        elif dur_sec > max_sec:
            # Too long: increase speed slightly
            current_speed = current_speed * 1.15
            continue
        else:
            trimmed = wav[0, : int(tts.sample_rate * dur_sec)]
            return trimmed, dur_sec
            
    # Fallback to trimmed audio
    trimmed = wav[0, : int(tts.sample_rate * dur[0].item())]
    final_dur = len(trimmed) / tts.sample_rate
    return trimmed, final_dur


def main():
    print("=== Supertonic-3 Japanese Voice Sample Generator ===")
    print(f"Loading ONNX models from {ONNX_DIR}...")
    tts = load_text_to_speech(ONNX_DIR)
    
    style_files = sorted([f for f in STYLES_DIR.glob("*.json") if f.name.lower() != "metadata.json"])
    print(f"Found {len(style_files)} style files in {STYLES_DIR}\n")
    
    metadata = []
    start_total = time.time()
    
    for idx, style_file in enumerate(style_files, 1):
        voice_key = style_file.stem
        if voice_key in CHARACTER_PROMPTS:
            text, speed = CHARACTER_PROMPTS[voice_key]
        else:
            # Generic fallback if an unrecognized style appears
            clean_name = voice_key.split("_", 1)[-1].replace("_", " ")
            text = f"はじめまして、{clean_name}です。私の声をお聞きいただき、心より感謝申し上げます。"
            speed = 0.86
            
        print(f"[{idx:02d}/{len(style_files)}] Synthesizing: {voice_key}")
        print(f"       Text: {text}")
        print(f"       Initial Speed: {speed}")
        
        t0 = time.time()
        voice_style = load_voice_style([str(style_file)])
        audio_data, duration = synthesize_with_duration_guard(tts, text, voice_style, speed)
        elapsed = time.time() - t0
        
        out_filename = f"{voice_key}.wav"
        out_path = OUTPUT_DIR / out_filename
        sf.write(str(out_path), audio_data, tts.sample_rate)
        
        status = "OK" if (5.0 <= duration <= 10.0) else "WARN"
        print(f"       -> Saved: {out_path.name} | Duration: {duration:.2f}s ({status}) in {elapsed:.2f}s\n")
        
        metadata.append({
            "index": idx,
            "voice_key": voice_key,
            "style_file": str(style_file),
            "output_audio": str(out_path),
            "text": text,
            "duration_seconds": round(duration, 3),
            "sample_rate": tts.sample_rate,
            "speed": speed,
            "status": status,
        })

    # Save metadata index
    meta_path = OUTPUT_DIR / "METADATA.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    total_elapsed = time.time() - start_total
    all_durations = [m["duration_seconds"] for m in metadata]
    all_ok = all(5.0 <= d <= 10.0 for d in all_durations)
    
    print("=" * 60)
    print(f"Generation Complete for {len(metadata)} voices!")
    print(f"Total time elapsed: {total_elapsed:.1f}s (avg {total_elapsed/len(metadata):.2f}s per voice)")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Metadata written: {meta_path.resolve()}")
    print(f"Duration range: {min(all_durations):.2f}s - {max(all_durations):.2f}s")
    print(f"All samples within 5-10s: {all_ok}")
    print("=" * 60)

if __name__ == "__main__":
    main()
