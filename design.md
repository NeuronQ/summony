```py
class ConnectorInterface:
    ...

class AgentInterface:
    ...

class ConversationInterface:
    ...

class UIInterface:
    ...

c = NBUI()#
c = NBUI(system_prompt='...', p_temperature=1.5)#

# c.set_system_prompt('...')

c.set_active_agents([agent_idx])#

#prams on ask/call

c('why is...') # aka c.ask()
c('explain me...', prefill='I will outline the steps to take first: ')#

c.set_params(...)#
c.set_params(0, ...)#

### reask
c() # aka c.reask()
c.to(agent_idx)
### ask single agent
c.to(agent_idx, 'why is...') # aka c.ask_single_agent()

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
```