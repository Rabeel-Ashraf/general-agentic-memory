# -*- coding: utf-8 -*-
"""
文本切分工具模块

提供多种文本切分策略：
- 按 token 数量切分
- 按句子切分
- 智能切分（结合段落和句子边界）
"""

import re
from typing import List, Optional
import tiktoken


def chunk_text_by_tokens(
    text: str, 
    max_tokens: int = 2000, 
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    按 token 数量切分文本
    
    Args:
        text: 待切分文本
        max_tokens: 每个块的最大 token 数
        encoding_name: tiktoken 编码名称
        
    Returns:
        切分后的文本块列表
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        # 如果无法加载编码，使用简单的字符切分
        return _chunk_by_chars(text, max_tokens * 4)  # 粗略估计
    
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks


def chunk_text_by_sentences(
    text: str,
    max_tokens: int = 2000,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    按句子边界切分文本，尽量保持句子完整
    
    Args:
        text: 待切分文本
        max_tokens: 每个块的最大 token 数
        encoding_name: tiktoken 编码名称
        
    Returns:
        切分后的文本块列表
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        return chunk_text_by_tokens(text, max_tokens, encoding_name)
    
    # 尝试使用 NLTK 进行句子分割
    sentences = _split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # 当前块已满，保存并开始新块
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def chunk_text_smartly(
    text: str,
    max_tokens: int = 2000,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    智能切分：结合段落和句子边界
    
    优先级：段落 > 句子 > token
    
    Args:
        text: 待切分文本
        max_tokens: 每个块的最大 token 数
        encoding_name: tiktoken 编码名称
        
    Returns:
        切分后的文本块列表
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        return chunk_text_by_tokens(text, max_tokens, encoding_name)
    
    # 先按段落分割
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_tokens = len(encoding.encode(para))
        
        # 如果段落本身超过限制，按句子切分
        if para_tokens > max_tokens:
            # 先保存当前块
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # 切分大段落
            sentences = _split_into_sentences(para)
            sent_chunk = []
            sent_tokens = 0
            
            for sent in sentences:
                sent_token_count = len(encoding.encode(sent))
                
                if sent_tokens + sent_token_count > max_tokens and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    sent_chunk = [sent]
                    sent_tokens = sent_token_count
                else:
                    sent_chunk.append(sent)
                    sent_tokens += sent_token_count
            
            if sent_chunk:
                chunks.append(" ".join(sent_chunk))
        
        elif current_tokens + para_tokens > max_tokens and current_chunk:
            # 当前块已满，保存并开始新块
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    # 添加最后一个块
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """
    将文本分割成句子
    
    优先使用 NLTK，如果不可用则使用正则表达式
    """
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        return nltk.sent_tokenize(text)
    except Exception:
        # 回退到简单的正则表达式分割
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def _chunk_by_chars(text: str, max_chars: int) -> List[str]:
    """简单的按字符数切分（回退方案）"""
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i:i + max_chars])
    return chunks

