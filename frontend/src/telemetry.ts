import { WebTracerProvider } from '@opentelemetry/sdk-trace-web'
import {
  SimpleSpanProcessor,
  ConsoleSpanExporter,
} from '@opentelemetry/sdk-trace-base'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto'
import { registerInstrumentations } from '@opentelemetry/instrumentation'
import { DocumentLoadInstrumentation } from '@opentelemetry/instrumentation-document-load'
import { resourceFromAttributes } from '@opentelemetry/resources'
import { ATTR_SERVICE_NAME } from '@opentelemetry/semantic-conventions'
import { ZoneContextManager } from '@opentelemetry/context-zone'

export function initializeTelemetry(
  otlpEndpoint: string,
  headers: string = '',
  resourceAttributes: string = '',
  serviceName: string = ''
) {
  const exporter = new OTLPTraceExporter({
    url: `${otlpEndpoint}/v1/traces`,
    headers: parseDelimitedValues(headers),
  })

  const attributes = parseDelimitedValues(resourceAttributes)

  attributes[ATTR_SERVICE_NAME] = serviceName?.trim() || 'browser'

  const provider = new WebTracerProvider({
    resource: resourceFromAttributes(attributes),
    spanProcessors: [
      new SimpleSpanProcessor(new ConsoleSpanExporter()),
      new SimpleSpanProcessor(exporter),
    ],
  })

  provider.register({
    contextManager: new ZoneContextManager(),
  })

  registerInstrumentations({
    instrumentations: [new DocumentLoadInstrumentation()],
  })
}

function parseDelimitedValues(value: string) {
  if (!value || !value.trim()) return {}
  return Object.fromEntries(
    value.split(',').map((pair) => {
      const [k, v] = pair.split('=')
      return [k.trim(), (v ?? '').trim()]
    })
  )
}
