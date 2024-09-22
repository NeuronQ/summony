class ConnectorInterface:
    ...

class AgentInterface:
    ...

class ConversationInterface:
    ...

class UIInterface:
    ...

c = NBUI()
c = NBUI(system_prompt='...', param_temperature=1.5)

c.set_system_prompt('...')

c.set_active_agents([agent_idx])

c('why is...')
c('explain me...', prefill='I will outline the steps to take first: ')

c.set_params(...)

c.alt('...')
c.alt()

c.select_alt(alt_idx)

message = {'id': '', 'role': '', 'content': '',
           # optional:
           'settings_idx': '', 'agent_id': ''}
converation = {'messages': [{}, {}, [{}, ...], ...],
               'last_messages': [2, 1],
               'settings': [{}, ...],
               'agents': [{}, ...],
               # optional:
               'active_agents_ids': [...]}

'''
a_conversations/
    conversation.json
    logs/  # full raw logs (<message_id>.json for each agent reply with inlined settings)
    files_sent/
    files_received/

'''