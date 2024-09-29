# TODO

- [X] ask and re-ask single agent (to param)
- conversation with (de)serialization in data format

# GOAL

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
c() # aka c.reask() -> ag.reask()
c(agent_idx) # aka c.reask(aidx) -> ag.reask()
### ask single agent
c.to(agent_idx, 'why is...') # aka c.ask()

# IMPORTANT: see actual code for latest structure

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
[
    Message.user,
    <OPTIONAL-PREFILL> Message.assistant </OPTIONAL-PREFILL>,
    {
        group_type: "replies",
        content: {
            <agent_idx>:
                | {group_type: "variants",
                   content: [Message.assistant, ...]},
                | {group_type: "thread",
                   content: [Message.assistant, Message.user, Message.assistant]},
            ...
        }
    },
    ...
]
'''

'''
a_conversations/
    conversation.json
    logs/  # full raw logs (<message_id>.json for each agent reply with inlined settings)
    files_sent/
    files_received/

'''
```
