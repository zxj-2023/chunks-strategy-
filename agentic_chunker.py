# 导入所需的库
from langchain_core.prompts import ChatPromptTemplate
import uuid
from langchain.chat_models import ChatOpenAI
import os
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains import create_extraction_chain_pydantic
from dotenv import load_dotenv

load_dotenv()

class AgenticChunker:
    """
    AgenticChunker 类负责将文本命题智能地分块。
    它使用大型语言模型（LLM）根据语义相关性将命题分组到“块”中。
    每个块都有一个动态生成的标题和摘要，随着更多命题的加入而更新。

    核心思想是模拟人类如何阅读和概念化长篇文档，即通过创建
    关于特定主题或思想的心理分组。

    属性:
        chunks (dict): 存储块的字典，以块ID为键。
        id_truncate_limit (int): 生成的UUID的截断长度。
        generate_new_metadata_ind (bool): 添加新命题时是否更新块元数据（摘要、标题）的标志。
        print_logging (bool): 是否在处理过程中打印日志的标志。
        llm (ChatOpenAI): 用于生成摘要、标题和进行相关性判断的语言模型实例。
    """
    def __init__(self, openai_api_key=None):
        """
        初始化 AgenticChunker。

        Args:
            openai_api_key (str, optional): OpenAI API 密钥。如果未提供，将从环境变量 `OPENAI_API_KEY` 中读取。

        Raises:
            ValueError: 如果 API 密钥既未提供也未在环境变量中找到。
        """
        self.chunks = {}
        self.id_truncate_limit = 5

        # Whether or not to update/refine summaries and titles as you get new information
        self.generate_new_metadata_ind = True
        self.print_logging = True

        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key is None:
            raise ValueError("API key is not provided and not found in environment variables")

        self.llm = ChatOpenAI(model='gpt-4-1106-preview', openai_api_key=openai_api_key, temperature=0)

    def add_propositions(self, propositions):
        """ 批量添加多个命题。 """
        for proposition in propositions:
            self.add_proposition(proposition)
    
    def add_proposition(self, proposition):
        """
        添加单个命题到合适的块中。

        这是核心逻辑，它决定一个命题是应该加入现有块还是创建一个新块。

        Args:
            proposition (str): 要添加的文本命题。
        """
        if self.print_logging:
            print (f"\nAdding: '{proposition}'")

        # If it's your first chunk, just make a new chunk and don't check for others
        if len(self.chunks) == 0:
            if self.print_logging:
                print ("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        # If a chunk was found then add the proposition to it
        if chunk_id:
            if self.print_logging:
                print (f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
            return
        else:
            if self.print_logging:
                print ("No chunks found")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(proposition)
        

    def add_proposition_to_chunk(self, chunk_id, proposition):
        """
        将命题添加到指定的块中，并选择性地更新该块的元数据。

        Args:
            chunk_id (str): 目标块的ID。
            proposition (str): 要添加的命题。
        """
        # Add then
        self.chunks[chunk_id]['propositions'].append(proposition)

        # Then grab a new summary
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        """
        （内部方法）当向块中添加新命题时，更新其摘要。

        这确保了摘要能够反映块中所有命题的最新、最全面的内容。

        Args:
            chunk (dict): 需要更新摘要的块。

        Returns:
            str: 新的、更新后的摘要。
        """
        """
        If you add a new proposition to a chunk, you may want to update the summary or else they could get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunks current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk new summary, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary']
        }).content

        return new_chunk_summary
    
    def _update_chunk_title(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the title or else it can get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
            ]
        )

        runnable = PROMPT | self.llm

        updated_chunk_title = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary'],
            "current_title" : chunk['title']
        }).content

        return updated_chunk_title

    def _get_new_chunk_summary(self, proposition):
        """
        （内部方法）为一个即将创建的新块生成初始摘要。

        摘要应具有通用性，以便将来可以容纳更多相关的命题。

        Args:
            proposition (str): 将要放入新块的第一个命题。

        Returns:
            str: 新块的摘要。
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the new chunk summary, nothing else.
                    """,
                ),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": proposition
        }).content

        return new_chunk_summary
    
    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

                    A good chunk title is brief but encompasses what the chunk is about

                    You will be given a summary of a chunk which needs a title

                    Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_title = runnable.invoke({
            "summary": summary
        }).content

        return new_chunk_title


    def _create_new_chunk(self, proposition):
        """
        （内部方法）基于一个初始命题创建一个全新的块。

        包括生成块ID、摘要和标题。

        Args:
            proposition (str): 新块的第一个命题。
        """
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit] # I don't want long ids
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id' : new_chunk_id,
            'propositions': [proposition],
            'title' : new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index' : len(self.chunks)
        }
        if self.print_logging:
            print (f"Created new chunk ({new_chunk_id}): {new_chunk_title}")
    
    def get_chunk_outline(self):
        """
        Get a string which represents the chunks you currently have.
        This will be empty when you first start off
        """
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ID: {chunk['chunk_id']}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"""
        
            chunk_outline += single_chunk_string
        
        return chunk_outline

    def _find_relevant_chunk(self, proposition):
        """
        （内部方法）判断一个新命题最适合放入哪个现有块。

        它将当前所有块的概要信息和新命题一起发送给LLM，
        让LLM判断是否存在语义匹配的块。

        Args:
            proposition (str): 要评估的命题。

        Returns:
            str or None: 如果找到匹配的块，则返回块ID；否则返回None。
        """
        current_chunk_outline = self.get_chunk_outline()

        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Determine whether or not the "Proposition" should belong to any of the existing chunks.

                    A proposition should belong to a chunk of their meaning, direction, or intention are similar.
                    The goal is to group similar propositions and chunks.

                    If you think a proposition should be joined with a chunk, return the chunk id.
                    If you do not think an item should be joined with an existing chunk, just return "No chunks"

                    Example:
                    Input:
                        - Proposition: "Greg really likes hamburgers"
                        - Current Chunks:
                            - Chunk ID: 2n4l3d
                            - Chunk Name: Places in San Francisco
                            - Chunk Summary: Overview of the things to do with San Francisco Places

                            - Chunk ID: 93833k
                            - Chunk Name: Food Greg likes
                            - Chunk Summary: Lists of the food and dishes that Greg likes
                    Output: 93833k
                    """,
                ),
                ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
                ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        chunk_found = runnable.invoke({
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        }).content

        # Pydantic data class
        class ChunkID(BaseModel):
            """Extracting the chunk id"""
            chunk_id: Optional[str]
            
        # Extraction to catch-all LLM responses. This is a bandaid
        extraction_chain = create_extraction_chain_pydantic(pydantic_schema=ChunkID, llm=self.llm)
        extraction_found = extraction_chain.run(chunk_found)
        if extraction_found:
            chunk_found = extraction_found[0].chunk_id

        # If you got a response that isn't the chunk id limit, chances are it's a bad response or it found nothing
        # So return nothing
        if len(chunk_found) != self.id_truncate_limit:
            return None

        return chunk_found
    
    def get_chunks(self, get_type='dict'):
        """
        This function returns the chunks in the format specified by the 'get_type' parameter.
        If 'get_type' is 'dict', it returns the chunks as a dictionary.
        If 'get_type' is 'list_of_strings', it returns the chunks as a list of strings, where each string is a proposition in the chunk.
        """
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['propositions']]))
            return chunks
    
    def pretty_print_chunks(self):
        print (f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print(f"Propositions:")
            for prop in chunk['propositions']:
                print(f"    -{prop}")
            print("\n\n")

    def pretty_print_chunk_outline(self):
        print ("Chunk Outline\n")
        print(self.get_chunk_outline())

if __name__ == "__main__":
    ac = AgenticChunker()

    ## Comment and uncomment the propositions to your hearts content
    propositions = [
        'The month is October.',
        'The year is 2023.',
        "One of the most important things that I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
        'Teachers and coaches implicitly told us that the returns were linear.',
        "I heard a thousand times that 'You get out what you put in.'",
        # 'Teachers and coaches meant well.',
        # "The statement that 'You get out what you put in' is rarely true.",
        # "If your product is only half as good as your competitor's product, you do not get half as many customers.",
        # "You get no customers if your product is only half as good as your competitor's product.",
        # 'You go out of business if you get no customers.',
        # 'The returns for performance are superlinear in business.',
        # 'Some people think the superlinear returns for performance are a flaw of capitalism.',
        # 'Some people think that changing the rules of capitalism would stop the superlinear returns for performance from being true.',
        # 'Superlinear returns for performance are a feature of the world.',
        # 'Superlinear returns for performance are not an artifact of rules that humans have invented.',
        # 'The same pattern of superlinear returns is observed in fame.',
        # 'The same pattern of superlinear returns is observed in power.',
        # 'The same pattern of superlinear returns is observed in military victories.',
        # 'The same pattern of superlinear returns is observed in knowledge.',
        # 'The same pattern of superlinear returns is observed in benefit to humanity.',
        # 'In fame, power, military victories, knowledge, and benefit to humanity, the rich get richer.'
    ]
    
    ac.add_propositions(propositions)
    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    print (ac.get_chunks(get_type='list_of_strings'))