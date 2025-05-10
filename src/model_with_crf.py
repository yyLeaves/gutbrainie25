from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchcrf import CRF


class BertCrfForTokenClassification(nn.Module):
    """
    结合了 BERT (或其他 Transformer) 和 CRF 层的 Token 分类模型。
    """
    def __init__(self, model_name_or_path, num_labels, dropout_prob=0.1):
        """
        初始化模型。

        Args:
            model_name_or_path (str): 预训练模型的名称或路径。
            num_labels (int): 标签的数量 (包括 B-, I-, O 标签)。
            dropout_prob (float, optional): Dropout 概率. Defaults to 0.1.
        """
        super().__init__()
        self.num_labels = num_labels

        # 加载基础 Transformer 模型配置，获取隐藏层大小等
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels # 传递 num_labels 以便 config 知晓
        )

        # 加载基础 Transformer 模型 (例如 BERT)，不使用其自带的池化层
        self.bert = AutoModel.from_pretrained(model_name_or_path, config=config, add_pooling_layer=False)

        # Dropout 层
        self.dropout = nn.Dropout(dropout_prob if dropout_prob is not None else config.hidden_dropout_prob)

        # 线性层：将 BERT 的输出映射到标签空间 (得到 CRF 的发射分数/emissions)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # CRF 层
        # 注意：CRF 层的 num_tags 需要和 classifier 输出维度以及标签数量一致
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, return_dict=False):
        """
        模型的前向传播。

        Args:
            input_ids (torch.Tensor): 输入 token IDs.
            attention_mask (torch.Tensor): 注意力掩码 (1 表示有效 token, 0 表示 padding).
            labels (torch.Tensor, optional): 真实的标签 ID. 训练时需要. Defaults to None.
            return_dict (bool, optional): 是否返回字典形式的输出. Defaults to False.

        Returns:
            loss (torch.Tensor) or decoded_tags (List[List[int]]):
                训练时返回 CRF 损失 (负对数似然)。
                预测时返回 Viterbi 解码后的最佳标签序列。
                如果 return_dict=True，则返回包含这些内容的字典。
        """
        # 1. 通过 BERT 获取序列输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)
        sequence_output = self.dropout(sequence_output)

        # 2. 通过线性分类器获取发射分数 (emissions/logits)
        emissions = self.classifier(sequence_output) # (batch_size, seq_length, num_labels)

        # 3. 根据是否有 labels 进行操作
        loss = None
        predictions = None
        mask = attention_mask.bool() # CRF 需要 boolean mask

        if labels is not None:
            # --- 训练阶段 ---
            # 使用 CRF 层计算损失 (负对数似然)
            # crf() 返回的是 log likelihood，我们需要其相反数作为损失
            labels_for_crf = labels.clone()
            o_label_id = 0
            labels_for_crf[labels == -100] = o_label_id
            loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction='mean')
        else:
            # --- 预测阶段 ---
            # 使用 CRF 层进行 Viterbi 解码
            predictions = self.crf.decode(emissions, mask=mask) # 返回 List[List[int]]
            

        # 4. 根据 return_dict 格式化输出
        if return_dict:
            output_dict = {"logits": emissions} # Logits (emissions) 总是可以返回
            if loss is not None:
                output_dict["loss"] = loss
            if predictions is not None:
                # 注意：CRF 返回的是 list of lists，不是 Tensor
                # 在使用 Accelerator 时，对非 Tensor 的 gather 需要特殊处理
                output_dict["predictions"] = predictions
            return output_dict
        else:
            # 旧的兼容方式：训练返回 loss，预测返回 predictions
            return loss if loss is not None else predictions

    def resize_token_embeddings(self, new_num_tokens=None):
        """
        辅助函数，用于在添加新 token 后调整 embedding 大小。
        调用底层 bert 模型的同名方法。
        """
        return self.bert.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, save_directory):
        """
        尝试兼容 Hugging Face 的 save_pretrained。
        保存 BERT 部分、分类器和 CRF 的状态字典以及配置文件。
        注意：CRF 层可能需要单独处理或确保能被 state_dict 捕获。
        更稳妥的方式可能是分别保存或只保存 state_dict。
        """
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        # 1. 保存 BERT 部分和配置文件
        self.bert.save_pretrained(save_directory)

        # 2. 保存分类器和 CRF 层的 state_dict
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'crf_state_dict': self.crf.state_dict()
            # 可以选择性保存 dropout 状态，但通常不需要
        }, path / "classifier_crf_state_dict.bin")

        print(f"BERT base saved to {save_directory}")
        print(f"Classifier and CRF state dict saved to {path / 'classifier_crf_state_dict.bin'}")

    # 可能还需要实现一个对应的 from_pretrained 类方法来加载，
    # 稍后再处理加载问题，先让训练跑起来。