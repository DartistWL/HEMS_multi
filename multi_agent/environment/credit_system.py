"""
Community Credit System
社区积分系统，实现动态定价机制
"""
import numpy as np


class CommunityCreditSystem:
    """
    社区积分系统
    管理家庭与社区共享储能的积分交易，使用动态定价机制
    """
    
    def __init__(self, n_agents=3, initial_credit=100.0):  # 默认初始积分为100.0
        """
        初始化积分系统
        
        Args:
            n_agents: 家庭数量
            initial_credit: 每个家庭的初始积分
        """
        self.n_agents = n_agents
        self.credit_balances = {i: initial_credit for i in range(n_agents)}
        self.initial_credit = initial_credit
        
        # 交易记录
        self.transaction_history = []
        
    def calculate_dynamic_price(self, grid_price, community_ess_soc, community_demand, 
                               transaction_type='buy',
                               agent_id=None,
                               pricing_scheme='uniform', contribution_discount_factor=0.0,
                               agent_energy_charged=None, max_energy_charged=None):
        """
        计算动态定价
        
        Args:
            grid_price: 当前电网电价
            community_ess_soc: 社区储能SOC（0-1）
            community_demand: 社区总需求（kW）
            transaction_type: 交易类型
                - 'buy': 从社区储能购买（放电）
                - 'sell': 向社区储能出售（充电）
            agent_id: 家庭ID（贡献定价时使用）
            pricing_scheme: 'uniform'=统一价格, 'contribution_based'=贡献多则购电更便宜（按向社区储能充电量）
            contribution_discount_factor: 贡献定价时购电折扣系数(0~1)，贡献越多折扣越大
            agent_energy_charged: 该户累计向社区储能充电量（kWh）
            max_energy_charged: 所有户中累计充电量最大值（用于归一化）
        
        Returns:
            动态价格（积分/kWh）
        """
        base_price = grid_price
        
        if transaction_type == 'buy':
            # 从社区储能购买：价格略高于电网价，鼓励使用社区储能
            # SOC越低，价格越高（鼓励节约使用）
            soc_factor = 1.0 + (1.0 - community_ess_soc) * 0.2  # SOC影响：0-20%
            # 需求越高，价格越高
            demand_factor = 1.0 + min(community_demand / 50.0, 0.3)  # 需求影响：0-30%
            price = base_price * 1.05 * soc_factor * (1.0 + demand_factor * 0.1)
            # 贡献定价：按向社区储能充电量，贡献多则购电更便宜
            if (pricing_scheme == 'contribution_based' and contribution_discount_factor > 0
                    and agent_energy_charged is not None and max_energy_charged is not None and max_energy_charged > 1e-9):
                relative_contribution = min(agent_energy_charged / max_energy_charged, 1.0)
                discount_ratio = contribution_discount_factor * max(0.0, relative_contribution)
                price = price * (1.0 - discount_ratio)
            
        else:  # 'sell'
            # 向社区储能出售：价格略低于电网价，鼓励贡献
            # SOC越低，收购价格越高（鼓励充电）
            soc_factor = 1.0 + (1.0 - community_ess_soc) * 0.3  # SOC影响：0-30%
            # 需求越高，收购价格越高
            demand_factor = 1.0 + min(community_demand / 50.0, 0.2)  # 需求影响：0-20%
            price = base_price * 0.95 * soc_factor * (1.0 + demand_factor * 0.1)
        
        return max(0.1, price)  # 确保价格为正
    
    def process_transaction(self, agent_id, energy_amount, price, transaction_type):
        """
        处理积分交易
        
        Args:
            agent_id: 家庭ID
            energy_amount: 能量数量（kWh）
            price: 交易价格（积分/kWh）
            transaction_type: 交易类型
                - 'buy': 从社区储能购买（消耗积分）
                - 'sell': 向社区储能出售（获得积分）
        
        Returns:
            dict: 交易结果
        """
        if energy_amount <= 0:
            return {
                'success': False,
                'message': 'Energy amount must be positive'
            }
        
        credit_change = energy_amount * price
        
        if transaction_type == 'buy':
            # 购买：消耗积分
            if self.credit_balances[agent_id] < credit_change:
                return {
                    'success': False,
                    'message': f'Insufficient credit: {self.credit_balances[agent_id]:.2f} < {credit_change:.2f}'
                }
            self.credit_balances[agent_id] -= credit_change
            credit_change = -credit_change  # 负值表示消耗
            
        else:  # 'sell'
            # 出售：获得积分
            self.credit_balances[agent_id] += credit_change
        
        # 记录交易
        transaction = {
            'agent_id': agent_id,
            'energy_amount': energy_amount,
            'price': price,
            'credit_change': credit_change,
            'transaction_type': transaction_type,
            'balance_after': self.credit_balances[agent_id]
        }
        self.transaction_history.append(transaction)
        
        return {
            'success': True,
            'credit_change': credit_change,
            'balance_after': self.credit_balances[agent_id],
            'transaction': transaction
        }
    
    def get_balance(self, agent_id):
        """
        获取家庭积分余额
        
        Args:
            agent_id: 家庭ID
        
        Returns:
            积分余额
        """
        return self.credit_balances.get(agent_id, 0.0)
    
    def get_all_balances(self):
        """
        获取所有家庭的积分余额
        
        Returns:
            dict: {agent_id: balance}
        """
        return self.credit_balances.copy()
    
    def reset(self):
        """
        重置积分系统
        """
        self.credit_balances = {i: self.initial_credit for i in range(self.n_agents)}
        self.transaction_history = []
